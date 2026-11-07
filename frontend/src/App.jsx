import { useState, useRef, useEffect } from 'react';
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';
import AppHeader from './components/AppHeader';
import GenerationForm from './components/GenerationForm';
import MidiEditor from './components/MidiEditor';
import { DRUM_SAMPLES, PRESET_DRUMS } from './constants/drums';

const API_BASE_URL = (import.meta.env.VITE_BACKEND_URL || '').replace(/\/$/, '');

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
  const [statusMessage, setStatusMessage] = useState('');

  const synthsRef = useRef([]);

  useEffect(() => {
    Tone.Transport.bpm.value = bpm;
  }, [bpm]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatusMessage('Running model inference & uploading to GCS...');

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
        headers: { 'Content-Type': 'application/json' },
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

  const loadMidiFromUrl = async (url) => {
    try {
      const midi = await Midi.fromUrl(url);
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

  const togglePlay = async () => {
    await Tone.start();
    if (isPlaying) {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      setIsPlaying(false);
      synthsRef.current.forEach(synth => synth.dispose());
      synthsRef.current = [];
    } else {
      if (!midiData || midiData.tracks.length === 0) {
        alert("No MIDI notes available to play.");
        return;
      }

      setIsPlaying(true);
      Tone.Transport.cancel();

      const sampler = new Tone.Sampler({
        urls: DRUM_SAMPLES,
        baseUrl: '/constants/',
        onload: () => {
          console.log('Sampler loaded successfully.');
        }
      }).toDestination();
      synthsRef.current.push(sampler);

      await Tone.loaded();
      
      midiData.tracks.forEach(track => {
        track.notes.forEach(note => {
          Tone.Transport.schedule((time) => {
            const preset = PRESET_DRUMS[note.name];
            if (preset) {
              sampler.attack = preset.envelope.attack;
              sampler.release = preset.envelope.release;
            }
            sampler.triggerAttackRelease(note.name, note.duration, time, note.velocity);
          }, note.time);
        });
      });

      Tone.Transport.start();
      
      const duration = midiData.duration + 1;
      setTimeout(() => {
        if (isPlaying) {
          setIsPlaying(false);
          Tone.Transport.stop();
        }
      }, duration * 1000);
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
      <AppHeader />

      <main className="w-full max-w-4xl">
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
            bpm={bpm}
            onBpmChange={setBpm}
            isPlaying={isPlaying}
            togglePlay={togglePlay}
            handleDownload={handleDownload}
            onBackToParams={() => setView('form')}
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