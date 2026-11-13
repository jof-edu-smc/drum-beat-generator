import { useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Download,
  MousePointerClick,
  Move,
  Play,
  Square
} from 'lucide-react';
import { DRUM_LANES } from '../constants/drums';

const SOURCE_BPM = 120;
const SIXTEENTH_NOTE = 60 / SOURCE_BPM / 4;

const snapToGrid = (time) => Math.max(0, Math.round(time / SIXTEENTH_NOTE) * SIXTEENTH_NOTE);

export default function MidiEditor({
  midiFilename,
  midiData,
  onDeleteNote,
  onAddNote,
  onUpdateNote,
  bpm,
  onBpmChange,
  isPlaying,
  playbackPosition,
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
  const [selectedNote, setSelectedNote] = useState(null);
  const [extensionWarning, setExtensionWarning] = useState('');
  const dragRef = useRef(null);
  const notes = useMemo(() => (
    midiData?.tracks.flatMap((track, trackIndex) =>
      track.notes.map((note, noteIndex) => ({ note, noteIndex, trackIndex }))
    ) ?? []
  ), [midiData]);
  const noteEnd = notes.reduce(
    (latest, { note }) => Math.max(latest, note.time + note.duration),
    0
  );
  const maxDuration = Math.max(midiData?.duration ?? 0, noteEnd, 4);
  const existingBars = Math.max(1, Math.ceil((midiData?.duration ?? 0) / 2));
  const selected = selectedNote
    ? midiData?.tracks[selectedNote.trackIndex]?.notes[selectedNote.noteIndex]
    : null;

  const laneNotes = useMemo(() => (
    DRUM_LANES.map((lane) => ({
      ...lane,
      notes: notes.filter(({ note }) => note.midi === lane.midi)
    }))
  ), [notes]);

  const beginNoteDrag = (event, noteInfo) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = event.currentTarget.parentElement.getBoundingClientRect();
    const edgeSize = Math.min(8, event.currentTarget.offsetWidth / 3);
    const localX = event.clientX - event.currentTarget.getBoundingClientRect().left;
    dragRef.current = {
      ...noteInfo,
      startX: event.clientX,
      initialTime: noteInfo.note.time,
      initialDuration: noteInfo.note.duration,
      pixelsPerSecond: bounds.width / maxDuration,
      mode: localX <= edgeSize ? 'resize-start' : (
        localX >= event.currentTarget.offsetWidth - edgeSize ? 'resize-end' : 'move'
      )
    };
    setSelectedNote(noteInfo);
  };

  const updateDraggedNote = (event) => {
    const drag = dragRef.current;
    if (!drag) return;
    const delta = (event.clientX - drag.startX) / drag.pixelsPerSecond;

    if (drag.mode === 'resize-start') {
      const end = drag.initialTime + drag.initialDuration;
      const time = Math.min(snapToGrid(drag.initialTime + delta), end - SIXTEENTH_NOTE);
      onUpdateNote(drag.trackIndex, drag.noteIndex, {
        time,
        duration: end - time
      });
    } else if (drag.mode === 'resize-end') {
      const duration = Math.max(
        SIXTEENTH_NOTE,
        snapToGrid(drag.initialDuration + delta)
      );
      onUpdateNote(drag.trackIndex, drag.noteIndex, { duration });
    } else {
      const targetTrack = document
        .elementFromPoint(event.clientX, event.clientY)
        ?.closest('.midi-roll__track');
      const targetMidi = targetTrack?.dataset.midi;
      onUpdateNote(drag.trackIndex, drag.noteIndex, {
        time: Math.min(snapToGrid(drag.initialTime + delta), maxDuration - SIXTEENTH_NOTE),
        ...(targetMidi ? { midi: Number(targetMidi) } : {})
      });
    }
  };

  const finishNoteDrag = () => {
    dragRef.current = null;
  };

  const handleExtendGenerate = () => {
    const requestedBars = Number(numBars);
    if (existingBars + requestedBars > 16) {
      const availableBars = 16 - existingBars;
      setExtensionWarning(
        availableBars > 0
          ? `This beat already contains ${existingBars} bars. Lower Number of Bars to ${availableBars} or fewer before extending.`
          : 'This beat already contains 16 bars. Start over to generate a new beat.'
      );
      return;
    }

    setExtensionWarning('');
    handleGenerate();
  };

  return (
    <div className="midi-editor bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-700 p-3">
        <div>
          <button onClick={onBackToParams} className="btn-secondary" type="button">
            <ArrowLeft className="w-4 h-4" /> Start Over
          </button>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={handleDownload} className="btn-primary" type="button">
            <Download className="w-4 h-4" /> {midiFilename}.mid
          </button>
        </div>
      </div>

      <section className="midi-editor__iteration">
        <div>
          <label htmlFor="editor-creativity">Adjust Creativity: {creativity}</label>
          <input
            id="editor-creativity"
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={creativity}
            onChange={(event) => setCreativity(event.target.value)}
          />
        </div>
        <div>
          <label htmlFor="editor-bars">Adjust Number of Bars: {numBars}</label>
          <input
            id="editor-bars"
            type="range"
            min="1"
            max="16"
            step="1"
            value={numBars}
            onChange={(event) => setNumBars(event.target.value)}
          />
        </div>
      </section>

      <section className="midi-editor__roll-section">
        <div className="midi-editor__instructions" role="note">
          <div><MousePointerClick /> <span><strong>Right-click</strong> a lane to add a note</span></div>
          <div><MousePointerClick /> <span><strong>Double-click</strong> a note to delete it</span></div>
          <div><Move /> <span><strong>Drag</strong> notes to move, resize, or change lanes</span></div>
        </div>
        {extensionWarning && (
          <div className="midi-editor__warning" role="alert">{extensionWarning}</div>
        )}
        <div className="midi-roll midi-roll--editor">
          <div className="midi-roll__timeline">
            {laneNotes.map((lane, laneIndex) => (
              <div
                key={lane.name}
                className="midi-roll__lane midi-roll__lane--rainbow"
                style={{ '--lane-hue': `${laneIndex * 17}` }}
              >
                <span className="midi-roll__label" title={`${lane.instrument} (${lane.name})`}>
                  {lane.instrument}
                </span>
                <div
                  className="midi-roll__track"
                  data-midi={lane.midi}
                  onContextMenu={(event) => {
                    event.preventDefault();
                    const bounds = event.currentTarget.getBoundingClientRect();
                    const time = snapToGrid(
                      ((event.clientX - bounds.left) / bounds.width) * maxDuration
                    );
                    onAddNote(lane.midi, Math.min(time, maxDuration - SIXTEENTH_NOTE));
                  }}
                >
                  {isPlaying && (
                    <span
                      className="midi-roll__play-cursor"
                      style={{ left: `${playbackPosition * 100}%` }}
                      aria-hidden="true"
                    />
                  )}
                  {lane.notes.map((noteInfo) => {
                    const leftPercent = (noteInfo.note.time / maxDuration) * 100;
                    const widthPercent = Math.max(
                      Math.min((noteInfo.note.duration / maxDuration) * 100, 100 - leftPercent),
                      0.6
                    );
                    const isSelected = selectedNote?.trackIndex === noteInfo.trackIndex
                      && selectedNote.noteIndex === noteInfo.noteIndex;

                    return (
                      <button
                        type="button"
                        key={`${noteInfo.trackIndex}-${noteInfo.noteIndex}-${noteInfo.note.time}`}
                        title={`${noteInfo.note.name} · velocity ${Math.round(noteInfo.note.velocity * 127)}`}
                        className={`midi-roll__note ${isSelected ? 'midi-roll__note--selected' : ''}`}
                        style={{ left: `${leftPercent}%`, width: `${widthPercent}%` }}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedNote(noteInfo);
                        }}
                        onDoubleClick={(event) => {
                          event.stopPropagation();
                          onDeleteNote(noteInfo.trackIndex, noteInfo.noteIndex);
                          setSelectedNote(null);
                        }}
                        onPointerDown={(event) => beginNoteDrag(event, noteInfo)}
                        onPointerMove={updateDraggedNote}
                        onPointerUp={finishNoteDrag}
                        onContextMenu={(event) => event.stopPropagation()}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          <button
            type="button"
            className="extend-generate"
            onClick={handleExtendGenerate}
            disabled={loading}
            title="Extend the generated beat"
            aria-label="Extend the generated beat"
          >
            <ArrowRight />
          </button>
        </div>
      </section>

      {selected && (
        <section className="velocity-control">
          <label htmlFor="velocity-slider">
            Selected velocity: {Math.round(selected.velocity * 127)}
          </label>
          <input
            id="velocity-slider"
            type="range"
            min="0"
            max="127"
            value={Math.round(selected.velocity * 127)}
            onChange={(event) => onUpdateNote(
              selectedNote.trackIndex,
              selectedNote.noteIndex,
              { velocity: Number(event.target.value) / 127 }
            )}
          />
        </section>
      )}

      <div className="playback-controls bg-slate-900 border border-slate-700/80 rounded-xl p-5 flex items-center gap-4">
        <button onClick={togglePlay} className={`p-4 rounded-full ${isPlaying ? 'bg-amber-600' : 'bg-emerald-600'} text-white`} type="button">
          {isPlaying ? <Square className="w-6 h-6" /> : <Play className="w-6 h-6" />}
        </button>
        <section className="bpm-control bpm-control--compact">
          <div className="bpm-control__header">
            <label htmlFor="bpm-slider">Tempo</label>
            <output htmlFor="bpm-slider">{bpm} BPM</output>
          </div>
          <input id="bpm-slider" type="range" min="40" max="240" value={bpm} onChange={(event) => onBpmChange(Number(event.target.value))} />
          <div className="bpm-control__scale" aria-hidden="true"><span>40</span><span>140</span><span>240</span></div>
        </section>

      </div>
    </div>
  );
}
