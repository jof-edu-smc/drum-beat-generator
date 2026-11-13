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
    url: "audio/kick.flac",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.35, release: 0.25 }
  },
  D2: {
    url: "audio/snare_head.flac",
    envelope: { attack: 0.001, decay: 0.12, sustain: 0.2, release: 0.18 }
  },
  "D#2": {
    url: "audio/snare_rim.flac",
    envelope: { attack: 0.001, decay: 0.08, sustain: 0.15, release: 0.12 }
  },
  E2: {
    url: "audio/snare_x_stick.flac",
    envelope: { attack: 0.001, decay: 0.1, sustain: 0.18, release: 0.15 }
  },
  F2: {
    url: "audio/tom_1_rim.flac",
    envelope: { attack: 0.001, decay: 0.3, sustain: 0.3, release: 0.3 }
  },
  "F#2": {
    url: "audio/tom_1.flac",
    envelope: { attack: 0.001, decay: 0.45, sustain: 0.35, release: 0.4 }
  },
  G2: {
    url: "audio/tom_2.flac",
    envelope: { attack: 0.001, decay: 0.4, sustain: 0.3, release: 0.35 }
  },
  "G#2": {
    url: "audio/tom_2_rim.flac",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.25, release: 0.2 }
  },
  A2: {
    url: "audio/tom_3.flac",
    envelope: { attack: 0.001, decay: 0.55, sustain: 0.35, release: 0.45 }
  },
  "A#2": {
    url: "audio/tom_3_rim.flac",
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.25, release: 0.2 }
  },
  B2: {
    url: "audio/hh_open_edge.flac",
    envelope: { attack: 0.001, decay: 0.4, sustain: 0.15, release: 0.35 }
  },
  C3: {
    url: "audio/hh_open_bow.flac",
    envelope: { attack: 0.001, decay: 0.45, sustain: 0.15, release: 0.4 }
  },
  "C#3": {
    url: "audio/hh_closed_bow.flac",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.08, sustain: 0.05, release: 0.08 }
  },
  D3: {
    url: "audio/hh_closed_edge.flac",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.06, sustain: 0.04, release: 0.06 }
  },
  "D#3": {
    url: "audio/hh_pedal.flac",
    envelope: { ...DEFAULT_ENVELOPE, decay: 0.12, sustain: 0.08, release: 0.1 }
  },
  E3: {
    url: "audio/crash_1_edge.flac",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.25, release: 0.8 }
  },
  F3: {
    url: "audio/crash_1_bow.flac",
    envelope: { attack: 0.001, decay: 0.9, sustain: 0.3, release: 0.9 }
  },
  "F#3": {
    url: "audio/crash_2_edge.flac",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.25, release: 0.8 }
  },
  G3: {
    url: "audio/crash_2_bow.flac",
    envelope: { attack: 0.001, decay: 0.9, sustain: 0.3, release: 0.9 }
  },
  "G#3": {
    url: "audio/ride_edge.flac",
    envelope: { attack: 0.001, decay: 0.7, sustain: 0.35, release: 0.65 }
  },
  A3: {
    url: "audio/ride_bow.flac",
    envelope: { attack: 0.001, decay: 0.8, sustain: 0.4, release: 0.7 }
  },
  "A#3": {
    url: "audio/ride_bell.flac",
    envelope: { attack: 0.001, decay: 0.6, sustain: 0.35, release: 0.55 }
  }
};

const GCS_AUDIO_BASE_URL = (
  import.meta.env.VITE_GCS_AUDIO_BASE_URL || ''
).replace(/\/$/, '');

export const DRUM_SAMPLES = {
  C2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/kick.flac` : '/audio/kick.flac',
  D2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/snare_head.flac` : '/audio/snare_head.flac',
  "D#2": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/snare_rim.flac` : '/audio/snare_rim.flac',
  E2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/snare_x_stick.flac` : '/audio/snare_x_stick.flac',
  F2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_1_rim.flac` : '/audio/tom_1_rim.flac',
  "F#2": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_1.flac` : '/audio/tom_1.flac',
  G2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_2.flac` : '/audio/tom_2.flac',
  "G#2": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_2_rim.flac` : '/audio/tom_2_rim.flac',
  A2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_3.flac` : '/audio/tom_3.flac',
  "A#2": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/tom_3_rim.flac` : '/audio/tom_3_rim.flac',
  B2: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/hh_open_edge.flac` : '/audio/hh_open_edge.flac',
  C3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/hh_open_bow.flac` : '/audio/hh_open_bow.flac',
  "C#3": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/hh_closed_bow.flac` : '/audio/hh_closed_bow.flac',
  D3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/hh_closed_edge.flac` : '/audio/hh_closed_edge.flac',
  "D#3": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/hh_pedal.flac` : '/audio/hh_pedal.flac',
  E3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/crash_1_edge.flac` : '/audio/crash_1_edge.flac',
  F3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/crash_1_bow.flac` : '/audio/crash_1_bow.flac',
  "F#3": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/crash_2_edge.flac` : '/audio/crash_2_edge.flac',
  G3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/crash_2_bow.flac` : '/audio/crash_2_bow.flac',
  "G#3": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/ride_edge.flac` : '/audio/ride_edge.flac',
  A3: GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/ride_bow.flac` : '/audio/ride_bow.flac',
  "A#3": GCS_AUDIO_BASE_URL ? `${GCS_AUDIO_BASE_URL}/ride_bell.flac` : '/audio/ride_bell.flac'
};

export const DRUM_INSTRUMENTS = {
  C2: 'Kick',
  D2: 'Snare Head',
  "D#2": 'Snare Rim',
  E2: 'Snare X-Stick',
  F2: 'Tom 1 Rim',
  "F#2": 'Tom 1',
  G2: 'Tom 2',
  "G#2": 'Tom 2 Rim',
  A2: 'Tom 3',
  "A#2": 'Tom 3 Rim',
  B2: 'Hi-Hat Open Edge',
  C3: 'Hi-Hat Open Bow',
  "C#3": 'Hi-Hat Closed Bow',
  D3: 'Hi-Hat Closed Edge',
  "D#3": 'Hi-Hat Pedal',
  E3: 'Crash 1 Edge',
  F3: 'Crash 1 Bow',
  "F#3": 'Crash 2 Edge',
  G3: 'Crash 2 Bow',
  "G#3": 'Ride Edge',
  A3: 'Ride Bow',
  "A#3": 'Ride Bell'
};

export const DRUM_LANES = Object.keys(PRESET_DRUMS).map((name) => {
  const [, pitch, octave] = name.match(/^([A-G]#?)(\d)$/);
  const midi = (Number(octave) + 1) * 12 + {
    C: 0,
    "C#": 1,
    D: 2,
    "D#": 3,
    E: 4,
    F: 5,
    "F#": 6,
    G: 7,
    "G#": 8,
    A: 9,
    "A#": 10,
    B: 11
  }[pitch];

  return {
    name,
    instrument: DRUM_INSTRUMENTS[name],
    midi,
    sample: DRUM_SAMPLES[name]
  };
});
