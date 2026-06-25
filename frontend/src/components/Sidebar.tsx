import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { getStrategies, getTickers, runBacktest, runSweep } from "../api/client";
import type {
  BacktestRequest,
  StrategyInfo,
  SweepRequest,
} from "../api/types";
import type { RunResult } from "../App";

type RunMode = "single" | "sweep";

interface SidebarProps {
  onResult: (r: RunResult) => void;
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function isoNYearsAgo(years: number): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - years);
  return d.toISOString().slice(0, 10);
}

// Build a `{key: [selected vals]}` map, defaulting to all values in `param_grid`.
function defaultGridSelection(
  grid: Record<string, unknown[]>,
): Record<string, unknown[]> {
  const out: Record<string, unknown[]> = {};
  for (const [k, vals] of Object.entries(grid)) {
    out[k] = [...vals];
  }
  return out;
}

export default function Sidebar({ onResult }: SidebarProps) {
  const tickersQuery = useQuery({ queryKey: ["tickers"], queryFn: getTickers });
  const strategiesQuery = useQuery({
    queryKey: ["strategies"],
    queryFn: getStrategies,
  });

  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [start, setStart] = useState<string>(isoNYearsAgo(2));
  const [end, setEnd] = useState<string>(todayIso());
  const [strategyName, setStrategyName] = useState<string>("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [gridSelection, setGridSelection] = useState<Record<string, unknown[]>>(
    {},
  );
  const [mode, setMode] = useState<RunMode>("single");

  // Auto-pick first strategy + load defaults once data arrives.
  useEffect(() => {
    if (!strategyName && strategiesQuery.data && strategiesQuery.data.length > 0) {
      const first = strategiesQuery.data[0];
      setStrategyName(first.name);
      setParams({ ...first.default_params });
      setGridSelection(defaultGridSelection(first.param_grid));
    }
  }, [strategiesQuery.data, strategyName]);

  const currentStrategy: StrategyInfo | undefined = useMemo(
    () => strategiesQuery.data?.find((s) => s.name === strategyName),
    [strategiesQuery.data, strategyName],
  );

  function onStrategyChange(name: string) {
    setStrategyName(name);
    const next = strategiesQuery.data?.find((s) => s.name === name);
    if (next) {
      setParams({ ...next.default_params });
      setGridSelection(defaultGridSelection(next.param_grid));
    }
  }

  function onParamChange(key: string, raw: string, isNumber: boolean) {
    setParams((prev) => ({
      ...prev,
      [key]: isNumber ? (raw === "" ? "" : Number(raw)) : raw,
    }));
  }

  function toggleTicker(t: string) {
    setSelectedTickers((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t],
    );
  }

  function toggleGridValue(key: string, val: unknown) {
    setGridSelection((prev) => {
      const cur = prev[key] ?? [];
      const has = cur.some((x) => x === val);
      const next = has ? cur.filter((x) => x !== val) : [...cur, val];
      return { ...prev, [key]: next };
    });
  }

  const sweepMutation = useMutation({
    mutationFn: (req: SweepRequest) => runSweep(req),
    onSuccess: (data, vars) => {
      const sweptKeys = Object.entries(vars.grid ?? {})
        .filter(([, vals]) => (vals as unknown[]).length > 1)
        .map(([k]) => k);
      onResult({ mode: "sweep", response: data, sweptKeys });
    },
    onError: (err) => {
      // eslint-disable-next-line no-console
      console.error("/sweep error:", err);
    },
  });

  const singleMutation = useMutation({
    mutationFn: (req: BacktestRequest) => runBacktest(req),
    onSuccess: (data) => {
      onResult({ mode: "single", response: data });
    },
    onError: (err) => {
      // eslint-disable-next-line no-console
      console.error("/backtest error:", err);
    },
  });

  const isPending = singleMutation.isPending || sweepMutation.isPending;
  const isError = singleMutation.isError || sweepMutation.isError;

  function onRun() {
    if (selectedTickers.length === 0 || !strategyName) {
      // eslint-disable-next-line no-console
      console.warn("Pick at least one ticker and a strategy before running.");
      return;
    }
    if (mode === "sweep") {
      // Strip empty axes — backend treats {} as "use defaults"; we want to send
      // only the axes the user explicitly selected.
      const grid: Record<string, unknown[]> = {};
      for (const [k, vals] of Object.entries(gridSelection)) {
        if (vals.length > 0) grid[k] = vals;
      }
      const req: SweepRequest = {
        tickers: selectedTickers,
        start,
        end,
        strategy: strategyName,
        grid,
      };
      sweepMutation.mutate(req);
      return;
    }
    const req: BacktestRequest = {
      tickers: selectedTickers,
      start,
      end,
      strategy: strategyName,
      params,
    };
    singleMutation.mutate(req);
  }

  return (
    <aside className="w-80 shrink-0 border-r border-slate-200 bg-slate-50 p-4 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold text-slate-800 mb-4">Backtest</h2>

      <section className="mb-5">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Tickers
        </label>
        {tickersQuery.isLoading && (
          <div className="text-xs text-slate-500">Loading…</div>
        )}
        {tickersQuery.isError && (
          <div className="text-xs text-red-600">Failed to load tickers.</div>
        )}
        {tickersQuery.data && tickersQuery.data.length === 0 && (
          <div className="text-xs text-slate-500">
            No tickers in ClickHouse yet.
          </div>
        )}
        {tickersQuery.data && tickersQuery.data.length > 0 && (
          <div className="max-h-40 overflow-y-auto border border-slate-200 rounded bg-white p-2 space-y-1">
            {tickersQuery.data.map((t) => (
              <label
                key={t}
                className="flex items-center gap-2 text-sm text-slate-700"
              >
                <input
                  type="checkbox"
                  checked={selectedTickers.includes(t)}
                  onChange={() => toggleTicker(t)}
                />
                <span className="font-mono">{t}</span>
              </label>
            ))}
          </div>
        )}
      </section>

      <section className="mb-5 grid grid-cols-2 gap-2">
        <label className="block text-sm font-medium text-slate-700 col-span-2">
          Date range
        </label>
        <input
          type="date"
          className="border border-slate-300 rounded px-2 py-1 text-sm"
          value={start}
          onChange={(e) => setStart(e.target.value)}
        />
        <input
          type="date"
          className="border border-slate-300 rounded px-2 py-1 text-sm"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
        />
      </section>

      <section className="mb-5">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Strategy
        </label>
        <select
          className="w-full border border-slate-300 rounded px-2 py-1 text-sm bg-white"
          value={strategyName}
          onChange={(e) => onStrategyChange(e.target.value)}
          disabled={!strategiesQuery.data}
        >
          {strategiesQuery.data?.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>
      </section>

      {currentStrategy && mode === "single" && (
        <section className="mb-5" data-testid="params-form">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Params
          </label>
          <div className="space-y-2">
            {Object.entries(currentStrategy.default_params).map(([key, defVal]) => {
              const isNumber = typeof defVal === "number";
              const value = params[key];
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="w-24 text-sm font-mono text-slate-600">
                    {key}
                  </span>
                  <input
                    type={isNumber ? "number" : "text"}
                    className="flex-1 border border-slate-300 rounded px-2 py-1 text-sm"
                    value={
                      value === undefined || value === null
                        ? ""
                        : String(value)
                    }
                    onChange={(e) => onParamChange(key, e.target.value, isNumber)}
                  />
                </div>
              );
            })}
          </div>
        </section>
      )}

      {currentStrategy && mode === "sweep" && (
        <section className="mb-5" data-testid="sweep-grid">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Sweep grid
          </label>
          <p className="text-xs text-slate-500 mb-2">
            Check the values to include for each param. Cartesian product of the
            checked values gets run.
          </p>
          <div className="space-y-3">
            {Object.entries(currentStrategy.param_grid).map(([key, vals]) => {
              const selected = gridSelection[key] ?? [];
              return (
                <div key={key}>
                  <div className="text-xs font-mono text-slate-600 mb-1">{key}</div>
                  <div className="flex flex-wrap gap-2">
                    {vals.map((v) => {
                      const isOn = selected.some((x) => x === v);
                      return (
                        <label
                          key={String(v)}
                          className={`px-2 py-0.5 text-xs rounded border cursor-pointer font-mono ${
                            isOn
                              ? "bg-indigo-600 text-white border-indigo-600"
                              : "bg-white text-slate-700 border-slate-300"
                          }`}
                        >
                          <input
                            type="checkbox"
                            className="sr-only"
                            checked={isOn}
                            onChange={() => toggleGridValue(key, v)}
                          />
                          {String(v)}
                        </label>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="mb-5">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Mode
        </label>
        <div className="flex items-center gap-4 text-sm text-slate-700">
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="mode"
              value="single"
              checked={mode === "single"}
              onChange={() => setMode("single")}
            />
            Single
          </label>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="mode"
              value="sweep"
              checked={mode === "sweep"}
              onChange={() => setMode("sweep")}
            />
            Sweep
          </label>
        </div>
      </section>

      <button
        onClick={onRun}
        disabled={isPending}
        className="w-full bg-indigo-600 text-white text-sm font-medium rounded px-3 py-2 hover:bg-indigo-700 disabled:opacity-50"
      >
        {isPending ? "Running…" : "Run"}
      </button>

      {isError && (
        <p className="text-xs text-red-600 mt-2">
          Run failed — check the console.
        </p>
      )}
    </aside>
  );
}
