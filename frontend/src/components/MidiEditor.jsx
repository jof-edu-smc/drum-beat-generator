import { Play, Square, Download, RefreshCw, CheckCircle, ArrowLeft } from 'lucide-react';

const DRUM_LANES = [
  { name: 'Ride Bell', midi: 53 },
  { name: 'Ride Bow', midi: 51 },
  { name: 'Crash 2', midi: 57 },
  { name: 'Crash 1', midi: 49 },
  { name: 'HH Pedal', midi: 44 },
  { name: 'HH Closed', midi: 42 },
  { name: 'HH Open', midi: 46 },
  { name: 'Tom 3', midi: 45 },
  { name: 'Tom 2', midi: 47 },
  { name: 'Tom 1', midi: 50 },
  { name: 'Snare', midi: 38 },
  { name: 'Kick', midi: 36 }
];

export default function MidiEditor({
  midiFilename,
  midiData,
  onDeleteNote,
  onAddNote,
  bpm,
  onBpmChange,
  isPlaying,
  togglePlay,
  handleDownload,
  onBackToParams,
  creativity,
  setCreativity,
  numBars,
  setNumBars,
  handleGenerate,
  loading
}) {
  const notes = midiData?.tracks.flatMap((track, trackIndex) =>
    track.notes.map((note, noteIndex) => ({ note, noteIndex, trackIndex }))
  ) ?? [];
  const noteEnd = notes.reduce(
    (latest, { note }) => Math.max(latest, note.time + note.duration),
    0
  );
  const maxDuration = Math.max(midiData?.duration ?? 0, noteEnd, 4);

  return (
    <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-6 shadow-xl space-y-6">
      
      {/* Top Action Header */}
      <div className="flex items-center justify-between border-b border-slate-700 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-emerald-400 flex items-center gap-2">
            <CheckCircle className="w-5 h-5" /> Generated: {midiFilename}.mid
          </h2>
          <p className="text-xs text-slate-400 mt-1">File Ready for Download.</p>
        </div>
        <button
          onClick={onBackToParams}
          className="bg-slate-700 hover:bg-slate-600 text-sm text-slate-200 font-medium px-4 py-2 rounded-lg transition flex items-center gap-2 shadow"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Parameters
        </button>
      </div>

      {/* Piano Roll Grid Inspector */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-slate-300">MIDI Track Piano Roll Inspector</h3>
        <div className="midi-roll">
          <div className="midi-roll__timeline">
            {DRUM_LANES.map((lane, laneIdx) => (
              <div key={laneIdx} className="midi-roll__lane">
                <span className="midi-roll__label">
                  {lane.name}
                </span>
                
                <div
                  className="midi-roll__track"
                  title={`Add ${lane.name} at the clicked time`}
                  onClick={(event) => {
                    const bounds = event.currentTarget.getBoundingClientRect();
                    const position = (event.clientX - bounds.left) / bounds.width;
                    const time = Math.min(Math.max(position * maxDuration, 0), maxDuration);
                    onAddNote(lane.midi, time);
                  }}
                >
                  {notes
                    .filter(({ note }) => note.midi === lane.midi)
                    .map(({ note, noteIndex, trackIndex }) => {
                      const leftPercent = (note.time / maxDuration) * 100;
                      const widthPercent = Math.max(
                        Math.min((note.duration / maxDuration) * 100, 100 - leftPercent),
                        0.5
                      );

                        return (
                          <button
                            type="button"
                            key={`${trackIndex}-${noteIndex}`}
                            title={`${note.name} | Time: ${note.time.toFixed(2)}s. Double-click to delete.`}
                            aria-label={`Delete ${note.name} at ${note.time.toFixed(2)} seconds`}
                            onClick={(event) => event.stopPropagation()}
                            onDoubleClick={() => onDeleteNote(trackIndex, noteIndex)}
                            className="midi-roll__note"
                            style={{
                              left: `${Math.min(Math.max(leftPercent, 0), 100)}%`,
                              width: `${widthPercent}%`
                            }}
                          />
                        );
                    })
                  }
                </div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-[11px] text-slate-500 italic">Showing Piano roll over a {maxDuration.toFixed(1)} second window.</p>
      </div>

      {/* Tempo Control */}
      <div className="bpm-control">
        <div className="bpm-control__header">
          <label htmlFor="bpm-slider">Tempo</label>
          <output htmlFor="bpm-slider">{bpm} BPM</output>
        </div>
        <input
          id="bpm-slider"
          type="range"
          min="40"
          max="240"
          step="1"
          value={bpm}
          onChange={(event) => onBpmChange(Number(event.target.value))}
          aria-label="Playback tempo in beats per minute"
        />
        <div className="bpm-control__scale" aria-hidden="true">
          <span>40</span>
          <span>140</span>
          <span>240</span>
        </div>
      </div>

      {/* Padded Audio Controls & Download Panel (Below Inspector) */}
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl p-5 mt-6 flex flex-col md:flex-row items-center justify-between gap-4 shadow-inner">
        <div className="flex items-center space-x-4">
          <button
            onClick={togglePlay}
            className={`p-4 rounded-full transition flex items-center justify-center ${
              isPlaying ? 'bg-amber-600 hover:bg-amber-500' : 'bg-emerald-600 hover:bg-emerald-500'
            } text-white shadow-md`}
          >
            {isPlaying ? <Square className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
          </button>
          <div>
            <h4 className="font-medium text-slate-200">Browser Audio Synthesizer</h4>
            <p className="text-xs text-slate-400">
              {isPlaying ? 'Playing piano roll sequence...' : 'Click play to audition generated MIDI notes.'}
            </p>
          </div>
        </div>

        <div>
          <button
            onClick={handleDownload}
            className="bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold px-5 py-3 rounded-xl shadow-lg transition flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Download MIDI File</span>
          </button>
        </div>
      </div>

      {/* Iteration Control Box */}
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
          className="w-full bg-slate-700 hover:bg-slate-600 text-white font-medium py-2.5 rounded-lg transition flex items-center justify-center space-x-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Re-run Generation with New Parameters</span>
        </button>
      </div>

    </div>
  );
}