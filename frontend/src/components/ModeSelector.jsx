import { RAG_MODES } from "../types";

export default function ModeSelector({ value, onChange }) {
  return (
    <fieldset className="panel">
      <legend>RAG architecture</legend>
      <div className="mode-grid">
        {RAG_MODES.map((mode) => (
          <label
            key={mode.id}
            className={`${value === mode.id ? "mode-card active" : "mode-card"}${mode.disabled ? " disabled" : ""}`}
          >
            <input
              type="radio"
              name="rag-mode"
              value={mode.id}
              checked={value === mode.id}
              disabled={mode.disabled}
              onChange={() => onChange(mode.id)}
            />
            <strong>{mode.label}</strong>
            <p>{mode.description}</p>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
