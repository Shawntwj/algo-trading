import { useState } from "react";

import Main from "./components/Main";
import Sidebar from "./components/Sidebar";
import type { BacktestResponse } from "./api/types";

export default function App() {
  const [lastResponse, setLastResponse] = useState<BacktestResponse | null>(null);

  return (
    <div className="flex h-screen w-screen bg-white text-slate-900">
      <Sidebar onResponse={setLastResponse} />
      <Main lastResponse={lastResponse} />
    </div>
  );
}
