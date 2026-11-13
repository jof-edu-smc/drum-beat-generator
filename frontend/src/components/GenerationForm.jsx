import { Sliders, Music, RefreshCw } from 'lucide-react';
import { GENRES } from '../constants/drums';

export default function GenerationForm({
  genre,
  setGenre,
  creativity,
  setCreativity,
  midiFilename,
  setMidiFilename,
  numBars,
  setNumBars,
  loading,
  statusMessage,
  onSubmit
}) {
  return (
    <form onSubmit={onSubmit} className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl space-y-6">
      <h2 className="text-xl font-semibold text-indigo-300 flex items-center gap-2">
        <Sliders className="w-5 h-5" /> Modify the parameters then generate a Drumbeat in MIDI
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">Select Genre</label>
          <select
            value={genre}
            onChange={(e) => setGenre(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-100 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            {GENRES.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>

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
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">MIDI Filename</label>
          <input
            type="text"
            value={midiFilename}
            onChange={(e) => setMidiFilename(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-100 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Number of Bars: <span className="text-indigo-400 font-bold">{numBars}</span>
          </label>
          <input
            type="range"
            min="1"
            max="16"
            step="1"
            value={numBars}
            onChange={(e) => setNumBars(e.target.value)}
            className="w-full accent-indigo-500 cursor-pointer mt-3"
          />
        </div>
      </div>

      <div className="pt-4 border-t border-slate-700 flex justify-end">
        <button
          type="submit"
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-3 rounded-lg shadow-lg transition flex items-center space-x-2 disabled:opacity-50"
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
    </form>
  );
}