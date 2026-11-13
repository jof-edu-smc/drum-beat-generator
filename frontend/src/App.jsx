import { useState, useRef, useEffect, useCallback } from 'react';
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';
import AppHeader from './components/AppHeader';
import AuthPanel from './components/AuthPanel';
import GenerationForm from './components/GenerationForm';
import MidiEditor from './components/MidiEditor';
import { DRUM_SAMPLES, PRESET_DRUMS } from './constants/drums';

const API_BASE_URL = (import.meta.env.VITE_BACKEND_URL || '').replace(/\/$/, '');
const MIDI_SOURCE_BPM = 120;

export default function App() {
  const [view, setView] = useState('form');
  const [genre, setGenre] = useState('All');
  const [creativity, setCreativity] = useState(1.0);
  const [midiFilename, setMidiFilename] = useState('generated_sample');
  const [numBars, setNumBars] = useState(4);
  const [bpm, setBpm] = useState(120);

  const [loading, setLoading] = useState(false);
  const [midiUrl, setMidiUrl] = useState('');
  const [midiData, setMidiData] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [authToken, setAuthToken] = useState('');
  const [user, setUser] = useState(null);

  const synthsRef = useRef([]);
  const transportStopRef = useRef(null);

  useEffect(() => {
    Tone.Transport.bpm.value = bpm;
  }, [bpm]);

  useEffect(() => {
    if (!isPlaying || !midiData?.duration) {
      return undefined;
    }

    let animationFrame;
    const updatePlaybackPosition = () => {
      setPlaybackPosition(Math.min(Tone.Transport.seconds / midiData.duration, 1));
      animationFrame = requestAnimationFrame(updatePlaybackPosition);
    };
    animationFrame = requestAnimationFrame(updatePlaybackPosition);

    return () => cancelAnimationFrame(animationFrame);
  }, [isPlaying, midiData]);

  const handleGenerate = async (e) => {
    e?.preventDefault?.();
    if (!authToken) {
      setStatusMessage('Sign in with Google before generating a beat.');
      return;
    }
    setLoading(true);
    setStatusMessage('Running model inference and prepping MIDI File for download...');

    try {
      const cleanedFilename = midiFilename.replace(/\s+/g, '_').trim();
      const payload = {
        genre: genre.toLowerCase(),
        creativity: parseFloat(creativity),
        midi_filename: cleanedFilename,
        num_bars: parseInt(numBars, 10),
        midi_url: midiUrl || ''
      };

      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error('Inference request failed on backend.');

      const data = await response.json();
      setMidiUrl(data.signed_url);
      await loadMidiFromUrl(data.signed_url);

      setLoading(false);
      setView('editor');
    } catch (err) {
      console.warn('Backend offline. Loading demo fallback structure...', err);
      setTimeout(() => {
        setLoading(false);
        const mockMidi = new Midi();
        const track = mockMidi.addTrack();
        track.addNote({ midi: 36, time: 0, duration: 0.25, velocity: 0.9 });
        track.addNote({ midi: 38, time: 1, duration: 0.25, velocity: 0.8 });
        track.addNote({ midi: 42, time: 0.5, duration: 0.1, velocity: 0.6 });
        setMidiData(mockMidi);
        setView('editor');
      }, 1000);
    }
  };

  const handleAuthenticated = useCallback((token, authenticatedUser) => {
    setAuthToken(token);
    setUser(authenticatedUser);
  }, []);

  const handleSignedOut = useCallback(() => {
    setAuthToken('');
    setUser(null);
  }, []);

  const onBackToParams = () => {
    setView('form');
    setMidiData(null);
    setMidiUrl('');
  }

  const loadMidiFromUrl = async (url) => {
    try {
      const separator = url.includes('?') ? '&' : '?';
      const midi = await Midi.fromUrl(`${url}${separator}v=${Date.now()}`);
      setMidiData(midi);
    } catch (e) {
      console.warn('Could not fetch remote MIDI, creating template.', e);
      setMidiData(new Midi());
    }
  };

  const handleDeleteNote = (trackIndex, noteIndex) => {
    if (!midiData?.tracks[trackIndex]?.notes[noteIndex]) return;

    const updatedMidi = midiData.clone();
    updatedMidi.tracks[trackIndex].notes.splice(noteIndex, 1);
    setMidiData(updatedMidi);
  };

  const handleAddNote = (midi, time) => {
    const updatedMidi = midiData ? midiData.clone() : new Midi();
    const track = updatedMidi.tracks[0] ?? updatedMidi.addTrack();

    track.addNote({
      midi,
      time,
      duration: 0.25,
      velocity: 0.8
    });
    setMidiData(updatedMidi);
  };

  const handleUpdateNote = (trackIndex, noteIndex, updates) => {
    if (!midiData?.tracks[trackIndex]?.notes[noteIndex]) return;

    const updatedMidi = midiData.clone();
    Object.assign(updatedMidi.tracks[trackIndex].notes[noteIndex], updates);
    setMidiData(updatedMidi);
  };

  const togglePlay = async () => {
    await Tone.start();

    if (isPlaying) {
      // If already playing, stop the transport and dispose of synths
      Tone.Transport.stop();
      Tone.Transport.cancel();
      transportStopRef.current?.();
    } else {

      if (!midiData || midiData.tracks.length === 0) {
        console.error("No MIDI notes available to play.");
        return;
      }

      setIsPlaying(true);
      Tone.Transport.cancel();
      const handleTransportStop = () => {
        Tone.Transport.off('stop', handleTransportStop);
        transportStopRef.current = null;

        setIsPlaying(false);
        setPlaybackPosition(0);
        synthsRef.current.forEach(synth => synth.dispose());
        synthsRef.current = [];
      };
      transportStopRef.current = handleTransportStop;
      Tone.Transport.on('stop', handleTransportStop);

      const sampler = new Tone.Sampler({ urls: DRUM_SAMPLES }).toDestination();
      synthsRef.current.push(sampler);

      try {
        await Tone.loaded();
      } catch (error) {
        Tone.Transport.cancel();
        Tone.Transport.off('stop', handleTransportStop);
        transportStopRef.current = null;
        sampler.dispose();
        synthsRef.current = [];
        setIsPlaying(false);
        console.error('Unable to load drum samples.', error);
        return;
      }
      
      midiData.tracks.forEach(track => {
        track.notes.forEach(note => {
          if (!Object.hasOwn(DRUM_SAMPLES, note.name)) {
            console.warn(`Skipping unsupported drum note: ${note.name} (MIDI ${note.midi})`);
            return;
          }

          const beatPosition = note.time * 2;
          const bar = Math.floor(beatPosition / 4);
          const beat = Math.floor(beatPosition % 4);
          const sixteenth = Math.round((beatPosition % 1) * 4);

          Tone.Transport.schedule((time) => {
            const preset = PRESET_DRUMS[note.name];
            if (preset) {
              sampler.attack = preset.envelope.attack;
              sampler.release = preset.envelope.release;
            }
            sampler.triggerAttackRelease(
              note.name,
              note.duration * (120 / bpm),
              time,
              note.velocity
            );
          }, `${bar}:${beat}:${sixteenth}`);
        });
      });

      const duration = midiData.duration + 1;
      Tone.Transport.scheduleOnce(() => {
        Tone.Transport.stop();
      }, duration * (MIDI_SOURCE_BPM / bpm));
      Tone.Transport.start();
    }
  };

  const handleDownload = () => {
    if (midiUrl) {
      const a = document.createElement('a');
      a.href = midiUrl;
      a.download = `${midiFilename}.mid`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else if (midiData) {
      const blob = new Blob([midiData.toArray()], { type: 'audio/midi' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${midiFilename}.mid`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center p-6">
      <AppHeader authControl={(
        <AuthPanel
          onAuthenticated={handleAuthenticated}
          onSignedOut={handleSignedOut}
          user={user}
        />
      )} />

      <main className={`w-full ${view === 'editor' ? 'editor-main' : 'max-w-4xl'}`}>
        {view === 'form' ? (
          <GenerationForm
            genre={genre}
            setGenre={setGenre}
            creativity={creativity}
            setCreativity={setCreativity}
            midiFilename={midiFilename}
            setMidiFilename={setMidiFilename}
            numBars={numBars}
            setNumBars={setNumBars}
            loading={loading}
            statusMessage={statusMessage}
            onSubmit={handleGenerate}
          />
        ) : (
          <MidiEditor
            midiFilename={midiFilename}
            midiData={midiData}
            onDeleteNote={handleDeleteNote}
            onAddNote={handleAddNote}
            onUpdateNote={handleUpdateNote}
            bpm={bpm}
            onBpmChange={setBpm}
            isPlaying={isPlaying}
            playbackPosition={playbackPosition}
            togglePlay={togglePlay}
            handleDownload={handleDownload}
            onBackToParams={onBackToParams}
            creativity={creativity}
            setCreativity={setCreativity}
            numBars={numBars}
            setNumBars={setNumBars}
            handleGenerate={handleGenerate}
            loading={loading}
          />
        )}
      </main>
    </div>
  );
}