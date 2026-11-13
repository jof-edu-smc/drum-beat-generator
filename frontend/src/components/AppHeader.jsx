import { Music } from 'lucide-react';

export default function AppHeader({ authControl }) {
  return (
    <header className="w-full max-w-4xl mb-8 flex items-center justify-between border-b border-slate-800 pb-4">
      <div className="flex items-center space-x-3">
        <Music className="w-8 h-8 text-indigo-400" />
        <h1 className="text-2xl font-bold tracking-tight">Drumbeat Studio</h1>
      </div>
      {authControl}
    </header>
  );
}