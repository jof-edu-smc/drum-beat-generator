import React, { useState, useEffect, useRef } from 'react';
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';
import { Play, Square, Download, RefreshCw, Sliders, Music, CheckCircle } from 'lucide-react';

const GENRES = [
  "All", "Funk", "Soul", "Hiphop", "Rock", "Jazz", 
  "NewOrleans", "Latin", "Afrocuban", "Punk"
];

export default function App() {
  // Navigation / View State: 'form' vs 'editor'
  const [view, setView] = useState('form');

  // Form Parameters
  const [genre, setGenre] = useState('All');
  const [creativity, setCreativity] = useState(1.0);
  const [midiFilename, setMidiFilename] = useState('generated_sample');
  const [numBars, setNumBars] = useState(4);

  // Application Status & Results
  const [loading, setLoading] = useState(false);
  const [midiUrl, setMidiUrl] = useState('');
  const [midiData, setMidiData] = useState(null); // Parsed MIDI object
  const [isPlaying, setIsPlaying] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [status, setStatus] = useState('ready');

  // Tone.js Synth / Player refs
  const synthsRef = useRef([]);

  // --- Step 1: Submit Generation Request to Backend API ---
  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatus('waiting');
    setStatusMessage('Running model inference & uploading to GCS...');

    try {
      // CLEAN FILENAME: replace whitespaces with underscores
      const cleanedFilename = midiFilename.replace(/\s+/g, '_').trim();

      const payload = {
        genre: genre.toLowerCase(),
        creativity: parseFloat(creativity),
        midi_filename: cleanedFilename,
        num_bars: parseInt(numBars, 10),
      };

      // REPLACE WITH YOUR BACKEND API ENDPOINT URL
      const response = await fetch('https://your-backend-api-url.run.app/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error('Inference request failed on backend.');

      const data = await response.json();
      // Expecting backend response to return { signed_url: 'https://storage.googleapis.com/...' }
      setMidiUrl(data.signed_url);

      // Fetch the binary MIDI file from the signed URL to parse into Tone.js Midi
      await loadMidiFromUrl(data.signed_url);

      setLoading(false);
      setStatus('ready');
      setView('editor'); // Move to Step 2: Editor view
    } catch (err) {
      console.error(err);
      // Fallback for local testing/demo if backend is offline: generate sample midi or mock url
      setStatus('error');
      setStatusMessage('Backend offline. Loading fallback demo structure...');
      setTimeout(() => {
        setLoading(false);
        setView('editor');
      }, 1000);
    }
  };

  // --- Step 2: Load and Parse MIDI into Tone.js ---
  const loadMidiFromUrl = async (url) => {
    try {
      const midi = await Midi.fromUrl(url);
      setMidiData(midi);
    } catch (e) {
      console.warn('Could not fetch remote MIDI, creating empty template structure.', e);
      setMidiData(new Midi());
    }
  };

  // --- Audio Playback using Tone.js ---
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

      // Create a polyphonic synth for playback
      const synth = new Tone.PolySynth(Tone.Synth).toDestination();
      synthsRef.current.push(synth);

      midiData.tracks.forEach(track => {
        track.notes.forEach(note => {
          Tone.Transport.schedule((time) => {
            synth.triggerAttackRelease(note.name, note.duration, time, note.velocity);
          }, note.time);
        });
      });

      Tone.Transport.start();
      
      // Auto-stop when finished
      const duration = midiData.duration + 1;
      setTimeout(() => {
        if (isPlaying) {
          setIsPlaying(false);
          Tone.Transport.stop();
        }
      }, duration * 1000);
    }
  };

  // --- Handle User Edits (e.g., adding/modifying note data interactively) ---
  const handleAddNoteMock = () => {
    if (!midiData) return;
    // Example interactive modification: clone or modify track notes
    const updatedMidi = midiData.clone();
    if (updatedMidi.tracks.length === 0) {
      updatedMidi.addTrack();
    }
    updatedMidi.tracks[0].addNote({
      midi: 36, // Standard Bass/Kick Drum pitch
      time: 0,
      duration: 0.25,
      velocity: 0.8
    });
    setMidiData(updatedMidi);
  };

  // --- Download Handler ---
  const handleDownload = () => {
    if (midiUrl) {
      // Option 1: Direct download from GCS signed URL
      const a = document.createElement('a');
      a.href = midiUrl;
      a.download = `${midiFilename}.mid`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else if (midiData) {
      // Option 2: Download local state buffer
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
      <header className="w-full max-w-4xl mb-8 flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <Music className="w-8 h-8 text-indigo-400" />
          <h1 className="text-2xl font-bold tracking-tight">MIDInet Studio Generator</h1>
        </div>
        <div className={`status-pill status-pill--${status}`} role="status" aria-live="polite">
          <span className="status-pill__dot" aria-hidden="true" />
          <span className="status-pill__label">
            {status === 'waiting' ? 'Waiting' : status === 'error' ? 'Error' : 'Ready'}
          </span>
        </div>
      </header>

      <main className="w-full max-w-4xl">
        {/* ================= VIEW 1: INITIAL PARAMETER FORM ================= */}
        {view === 'form' && (
          <form onSubmit={handleGenerate}>
            <div className="generation-parameters">
              <div className="generation-parameters__header">
                <div>
                  <p className="generation-parameters__eyebrow">MIDI composition</p>
                  <h2 className="text-xl font-semibold text-indigo-300 flex items-center gap-2">
                    <Sliders className="w-5 h-5" /> Generation Parameters
                  </h2>
                </div>
                <span className="generation-parameters__badge">Configure</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 1. Genre Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Select Genre</label>
                <select
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                  className="form-select"
                >
                  {GENRES.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
                <p className="text-xs text-slate-500 mt-1">Filters dataset weights or pre-trained checkpoints.</p>
              </div>

              {/* 2. Level of Creativity Slider */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Level of Creativity (Noise Multiplier): <span className="text-indigo-400 font-bold">{creativity}</span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={creativity}
                  onChange={(e) => setCreativity(e.target.value)}
                  className="w-full accent-indigo-500 cursor-pointer mt-3"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <ul>
                        <li>0.5: Conservative</li>
                        <li>1.0: Balanced</li>
                        <li>2.0: Wild</li>
                    </ul>
                </div>
              </div>

              {/* 3. MIDI Filename */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">MIDI Filename</label>
                <input
                  type="text"
                  value={midiFilename}
                  onChange={(e) => setMidiFilename(e.target.value)}
                  placeholder="generated_sample"
                  className="form-input"
                />
                <p className="text-xs text-slate-500 mt-1">Spaces will automatically be converted to underscores.</p>
              </div>

              {/* 4. Number of Bars */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Number of Bars: <span className="text-indigo-400 font-bold">{numBars}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="32"
                  step="1"
                  value={numBars}
                  onChange={(e) => setNumBars(e.target.value)}
                  className="w-full accent-indigo-500 cursor-pointer mt-3"
                />
                <p className="text-xs text-slate-500 mt-1">Identifies 4/4 musical bars length.</p>
              </div>
              </div>

              <div className="pt-4 border-t border-slate-700 flex justify-end">
                <button
                  type="submit"
                  disabled={loading}
                  className="btn btn-primary"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      <span>Processing Inference...</span>
                    </>
                  ) : (
                    <>
                      <Music className="w-5 h-5" />
                      <span>Generate MIDI Track</span>
                    </>
                  )}
                </button>
              </div>
              {loading && <p className="text-center text-sm text-indigo-400 animate-pulse">{statusMessage}</p>}
            </div>
          </form>
        )}

        {/* ================= VIEW 2: EDITABLE MIDI EDITOR & GENERATION RESPONSE ================= */}
        {view === 'editor' && (
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl space-y-6">
            <div className="flex items-center justify-between border-b border-slate-700 pb-4">
              <div>
                <h2 className="text-xl font-semibold text-emerald-400 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" /> Generation Complete: {midiFilename}.mid
                </h2>
                <p className="text-xs text-slate-400 mt-1">Stored securely via Cloud Storage Signed URL.</p>
              </div>
              <button
                onClick={() => setView('form')}
                className="btn btn-secondary btn-small"
              >
                ← Back to Parameters
              </button>
            </div>

            {/* Playback Controls & Note Inspection */}
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-5 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center space-x-4">
                <button
                  onClick={togglePlay}
                  className={`btn btn-play ${isPlaying ? 'playing' : ''}`}
                >
                  {isPlaying ? <Square className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
                </button>
                <div>
                  <h4 className="font-medium text-slate-200">Browser Audio Synthesizer</h4>
                  <p className="text-xs text-slate-400">
                    {isPlaying ? 'Playing sequence...' : 'Click play to audition generated MIDI notes.'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleAddNoteMock}
                  className="btn btn-secondary btn-small"
                >
                  + Add Drum Note (Edit)
                </button>
                <button
                  onClick={handleDownload}
                  className="btn btn-primary btn-small"
                >
                  <Download className="w-4 h-4" />
                  <span>Download MIDI</span>
                </button>
              </div>
            </div>

            {/* Interactive Note Inspector / Editor View */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-300">Parsed MIDI Track Inspector (Tone.js / @tonejs/midi):</h3>
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 max-h-60 overflow-y-auto font-mono text-xs text-slate-400 space-y-1">
                {midiData && midiData.tracks.length > 0 ? (
                  midiData.tracks[0].notes.map((note, idx) => (
                    <div key={idx} className="flex justify-between border-b border-slate-800/50 py-1">
                      <span>Note: <strong className="text-indigo-300">{note.name}</strong> (MIDI: {note.midi})</span>
                      <span>Time: {note.time.toFixed(2)}s</span>
                      <span>Duration: {note.duration.toFixed(2)}s</span>
                      <span>Velocity: {note.velocity.toFixed(2)}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500 italic">No notes found or demo mode active.</p>
                )}
              </div>
            </div>

            {/* Iterative Refinement Section (Trigger next generation with updated params) */}
            <div className="border-t border-slate-700 pt-6">
              <h3 className="text-sm font-semibold text-slate-200 mb-4">Iterate & Re-trigger Generation Sequence</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Adjust Creativity: {creativity}</label>
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.1"
                    value={creativity}
                    onChange={(e) => setCreativity(e.target.value)}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Adjust Number of Bars: {numBars}</label>
                  <input
                    type="range"
                    min="1"
                    max="32"
                    step="1"
                    value={numBars}
                    onChange={(e) => setNumBars(e.target.value)}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              </div>
              <button
                onClick={handleGenerate}
                disabled={loading}
                className="btn btn-secondary btn-full"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span>Re-run Generation with New Parameters</span>
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}