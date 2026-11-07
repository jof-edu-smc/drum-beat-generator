export const GENRES = [
  "All", "Funk", "Soul", "Hiphop", "Rock", "Jazz",
  "NewOrleans", "Latin", "Afrocuban", "Punk"
];

const DEFAULT_ENVELOPE = {
  attack: 0.001,
  decay: 0.15,
  sustain: 0.25,
  release: 0.2
};

export const PRESET_DRUMS = {
  C2: {
    url: "audio/kick.wav",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.35, release: 0.25 }
  },
  D2: {
    url: "audio/snare_head.wav",
    envelope: { attack: 0.001, decay: 0.12, sustain: 0.2, release: 0.18 }
  },
  "D#2": {
    url: "audio/snare_rim.wav",
    envelope: { attack: 0.001, decay: 0.08, sustain: 0.15, release: 0.12 }
  },
  E2: {
    url: "audio/snare_x_stick.wav",
    envelope: { attack: 0.001, decay: 0.1, sustain: 0.18, release: 0.15 }
  },
  F2: {
    url: "audio/tom_1_rim.wav",
    envelope: { attack: 0.001, decay: 0.3, sustain: 0.3, release: 0.3 }
  },
  "F#2": {
    url: "audio/tom_1.wav",
    envelope: { attack: 0.001, decay: 0.45, sustain: 0.35, release: 0.4 }
  },
  G2: {
    url: "audio/tom_2.wav",
    envelope: { attack: 0.001, decay: 0.4, sustain: 0.3, release: 0.35 }
  },
  "G#2": {
    url: "audio/tom_2_rim.wav",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.25, release: 0.2 }
  },
  A2: {
    url: "audio/tom_3_head.wav",
    envelope: { attack: 0.001, decay: 0.55, sustain: 0.35, release: 0.45 }
  },
  "A#2": {
    url: "audio/tom_3_rim.wav",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.25, release: 0.2 }
  },
  B2: {
    url: "audio/hh_open_edge.wav",
    envelope: { attack: 0.001, decay: 0.4, sustain: 0.15, release: 0.35 }
  },
  C3: {
    url: "audio/hh_open_bow.wav",
    envelope: { attack: 0.001, decay: 0.45, sustain: 0.15, release: 0.4 }
  },
  "C#3": {
    url: "audio/hh_closed_bow.wav",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.08, sustain: 0.05, release: 0.08 }
  },
  D3: {
    url: "audio/hh_closed_edge.wav",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.06, sustain: 0.04, release: 0.06 }
  },
  "D#3": {
    url: "audio/hh_pedal.wav",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.12, sustain: 0.08, release: 0.1 }
  },
  E3: {
    url: "audio/crash_1_edge.wav",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.25, release: 0.8 }
  },
  F3: {
    url: "audio/crash_1_bow.wav",
    envelope: { attack: 0.001, decay: 0.9, sustain: 0.3, release: 0.9 }
  },
  "F#3": {
    url: "audio/crash_2_edge.wav",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.25, release: 0.8 }
  },
  G3: {
    url: "audio/crash_2_bow.wav",
    envelope: { attack: 0.001, decay: 0.9, sustain: 0.3, release: 0.9 }
  },
  "G#3": {
    url: "audio/ride_edge.wav",
    envelope: { attack: 0.001, decay: 0.7, sustain: 0.35, release: 0.65 }
  },
  A3: {
    url: "audio/ride_bow.wav",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.4, release: 0.7 }
  },
  "A#3": {
    url: "audio/ride_bell.wav",
    envelope: { attack: 0.001, decay: 0.6, sustain: 0.35, release: 0.55 }
  }
};

export const DRUM_SAMPLES = Object.fromEntries(
  Object.entries(PRESET_DRUMS).map(([note, preset]) => [note, preset.url])
);
