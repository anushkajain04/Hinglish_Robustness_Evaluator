import React, { useState, useEffect } from 'react';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [modelType, setModelType] = useState('upgraded'); // 'upgraded' | 'baseline'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);

  // Fetch model benchmark metrics on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/metrics')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load metrics');
        return res.json();
      })
      .then((data) => setMetrics(data))
      .catch((err) => {
        console.warn('Metrics endpoint not reachable yet:', err);
      });
  }, []);

  const handleAnalyze = async (e) => {
    if (e && e.preventDefault) {
      e.preventDefault();
    }
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sentence: inputText,
          model_type: modelType,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error analyzing sentence:', err);
      setError(
        err.message || 'Unable to connect to analysis server. Please verify backend is running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const sampleSentences = [
    { text: 'Yeh movie bohot achhi thi yaar', label: 'Positive Slang' },
    { text: 'm aur mjhe ye movie bht achi lgi yr, par dost ko blkl bkwas lgi', label: 'SMS Shorthand' },
    { text: 'Khana bilkul bhi achha nahi tha bakwas', label: 'SOV Negation' },
    { text: 'bhai ekdum bkwass movie thi,paisa brbad', label: 'Subword Noise' },
    { text: 'Service theek thak thi kuch khaas nahi', label: 'Neutral/Negation' },
  ];

  // Helper to render negation tokens
  const renderNegationTokens = (taggedText) => {
    if (!taggedText) return null;
    const tokens = taggedText.split(/\s+/);
    return (
      <div className="flex flex-wrap items-center gap-1.5 mt-2">
        {tokens.map((token, idx) => {
          const isNegated = token.startsWith('NOT_');
          return (
            <span
              key={idx}
              className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                isNegated
                  ? 'bg-amber-100 text-amber-900 border border-amber-300 font-bold shadow-xs'
                  : 'bg-slate-100 text-slate-700 border border-slate-200'
              }`}
              title={isNegated ? 'Negation scope attached' : undefined}
            >
              {token}
            </span>
          );
        })}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 antialiased font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-sm">
              HR
            </div>
            <div>
              <span className="font-semibold text-slate-900 tracking-tight block sm:inline">
                Hinglish Robustness Suite
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100">
                v2.0 Classical Upgrade
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-xs font-medium text-slate-500">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>API: 127.0.0.1:8000</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Header Section */}
        <div className="text-center max-w-2xl mx-auto mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-100 mb-4">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            NLP Robustness & Perturbation Benchmark
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Hinglish Robustness Evaluator
          </h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 leading-relaxed">
            Assess NLP prediction consistency against phonetic perturbations, character noise, and code-mixed SOV negation patterns.
          </p>
        </div>

        {/* Model Benchmark Card */}
        {metrics && (
          <section className="mb-6 bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  Hold-out Test Benchmark (80-20 Stratified Split)
                </h2>
              </div>
              <span className="text-xs text-slate-400">
                Test Set: {metrics.upgraded?.test_samples || 643} samples
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Baseline Metrics */}
              <div className={`p-3.5 rounded-xl border transition-all ${
                modelType === 'baseline'
                  ? 'border-indigo-400 bg-indigo-50/40 ring-2 ring-indigo-200'
                  : 'border-slate-200 bg-slate-50/50'
              }`}>
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span>Baseline (Word TF-IDF + LogReg)</span>
                  {modelType === 'baseline' && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-indigo-600 text-white font-bold">Active</span>
                  )}
                </div>
                <div className="flex items-center space-x-4 mt-2">
                  <div>
                    <span className="text-[11px] text-slate-500 block">Test Accuracy</span>
                    <span className="text-base font-bold text-slate-800">{metrics.baseline?.accuracy}%</span>
                  </div>
                  <div className="border-l border-slate-200 pl-4">
                    <span className="text-[11px] text-slate-500 block">Macro F1</span>
                    <span className="text-base font-bold text-slate-800">{metrics.baseline?.macro_f1}%</span>
                  </div>
                </div>
              </div>

              {/* Upgraded Metrics */}
              <div className={`p-3.5 rounded-xl border transition-all ${
                modelType === 'upgraded'
                  ? 'border-emerald-500 bg-emerald-50/40 ring-2 ring-emerald-200'
                  : 'border-slate-200 bg-slate-50/50'
              }`}>
                <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-1">
                  <span className="flex items-center space-x-1.5">
                    <span>Classical Upgrade (LinearSVC + Negation)</span>
                    <span className="px-1.5 py-0.2 rounded text-[10px] bg-emerald-100 text-emerald-800 font-bold border border-emerald-300">
                      {metrics.upgraded && metrics.baseline
                        ? `+${(metrics.upgraded.macro_f1 - metrics.baseline.macro_f1).toFixed(2)}% F1`
                        : '+1.03% F1'}
                    </span>
                  </span>
                  {modelType === 'upgraded' && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-600 text-white font-bold">Active</span>
                  )}
                </div>
                <div className="flex items-center space-x-4 mt-2">
                  <div>
                    <span className="text-[11px] text-slate-500 block">Test Accuracy</span>
                    <span className="text-base font-bold text-emerald-700">{metrics.upgraded?.accuracy}%</span>
                  </div>
                  <div className="border-l border-slate-200 pl-4">
                    <span className="text-[11px] text-slate-500 block">Macro F1</span>
                    <span className="text-base font-bold text-emerald-700">{metrics.upgraded?.macro_f1}%</span>
                  </div>
                  <div className="border-l border-slate-200 pl-4">
                    <span className="text-[11px] text-slate-500 block">Neutral F1</span>
                    <span className="text-base font-bold text-emerald-700">54% (+8%)</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Input Area Card */}
        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 sm:p-8 mb-8 transition-all">
          <form onSubmit={handleAnalyze}>
            {/* Model Selector Segmented Control */}
            <div className="mb-5">
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                Select Model Pipeline
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-1 bg-slate-100 rounded-xl">
                <button
                  type="button"
                  onClick={() => setModelType('upgraded')}
                  className={`flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all ${
                    modelType === 'upgraded'
                      ? 'bg-white text-emerald-700 shadow-sm border border-slate-200'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <span className="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
                  <span>Classical Upgrade (LinearSVC + Negation)</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Recommended
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => setModelType('baseline')}
                  className={`flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-all ${
                    modelType === 'baseline'
                      ? 'bg-white text-indigo-700 shadow-sm border border-slate-200'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <span className="inline-block w-2 h-2 rounded-full bg-indigo-500"></span>
                  <span>Baseline (Word TF-IDF + LogReg)</span>
                </button>
              </div>
            </div>

            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="hinglish-input" className="block text-sm font-semibold text-slate-700">
                  Input Sentence (Hinglish)
                </label>
                <span className="text-xs text-slate-400">
                  {inputText.length} characters
                </span>
              </div>
              <textarea
                id="hinglish-input"
                rows={4}
                className="w-full rounded-xl border border-slate-200 bg-slate-50/50 p-4 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 outline-none transition duration-150 resize-y text-base"
                placeholder="e.g. Bohat acchi movie thi yaar, full paisa vasool!"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                disabled={loading}
              />
            </div>

            {/* Quick Test Samples */}
            <div className="flex flex-wrap items-center gap-2 mb-6">
              <span className="text-xs font-medium text-slate-400">Try sample:</span>
              {sampleSentences.map((sample, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setInputText(sample.text)}
                  disabled={loading}
                  className="text-xs px-2.5 py-1 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition flex items-center space-x-1.5"
                >
                  <span>"{sample.text}"</span>
                  <span className="text-[10px] text-slate-400">({sample.label})</span>
                </button>
              ))}
            </div>

            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="text-xs text-slate-500">
                Testing against: <strong className="text-slate-700 font-semibold">{modelType === 'upgraded' ? 'Classical Upgrade Pipeline' : 'Baseline Pipeline'}</strong>
              </div>
              <button
                type="submit"
                disabled={loading || !inputText.trim()}
                className="w-full sm:w-auto inline-flex items-center justify-center px-6 py-3 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm shadow-indigo-200 transition-all duration-150"
              >
                {loading ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2.5 h-4 w-4 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v8H4z"
                      />
                    </svg>
                    Analyzing Robustness...
                  </>
                ) : (
                  'Test Robustness'
                )}
              </button>
            </div>
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 rounded-xl bg-red-50 border border-red-200 text-sm text-red-700 flex items-start space-x-3">
              <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="font-medium">Request Failed</p>
                <p className="text-xs mt-0.5 text-red-600">{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* Results Section */}
        {result && (
          <div className="space-y-8 animate-fadeIn">
            {/* Model Badge & Preprocessing Info */}
            <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Model Evaluated:</span>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${
                  result.model_used && result.model_used.includes('Upgrade')
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                }`}>
                  {result.model_used || (modelType === 'upgraded' ? 'Classical Upgrade' : 'Baseline')}
                </span>
              </div>

              {result.negation_tagged && (
                <div className="text-xs text-slate-500 flex items-center space-x-1">
                  <span>Negation Tagging:</span>
                  <span className="font-medium text-slate-700">Bidirectional SOV Active</span>
                </div>
              )}
            </div>

            {/* Negation-Tagged Preview Box */}
            {result.negation_tagged && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      Hinglish Negation Scope Transformation
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200 font-semibold">
                      NOT_ Prefix Binding
                    </span>
                  </div>
                  <span className="text-xs text-slate-400">Custom Scikit-Learn Transformer</span>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Standard n-grams fail when negation particles (<em>nahi, mat, na</em>) appear after verbs in Hindi/Hinglish (SOV order). The Negation Tagger binds negation across a window of 2 words in both directions:
                </p>
                {renderNegationTokens(result.negation_tagged)}
              </div>
            )}

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Baseline Prediction Card */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase">
                      Sentence Prediction
                    </span>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
                      Original Input
                    </span>
                  </div>
                  <p className="mt-3 text-base text-slate-800 font-medium italic bg-slate-50 p-3 rounded-lg border border-slate-100">
                    "{result.original}"
                  </p>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
                  <span className="text-sm text-slate-500 font-medium">Predicted Sentiment:</span>
                  <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border ${
                    result.baseline_pred === 'Positive'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : result.baseline_pred === 'Negative'
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'
                  }`}>
                    {result.baseline_pred}
                  </span>
                </div>
              </div>

              {/* Consistency Score Card */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase">
                    Consistency Score
                  </span>
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                      result.consistency_score >= 50
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : 'bg-red-50 text-red-700 border border-red-200'
                    }`}
                  >
                    {result.consistency_score >= 50 ? 'Stable' : 'Vulnerable'}
                  </span>
                </div>

                <div className="mt-4 flex items-baseline space-x-2">
                  <span
                    className={`text-5xl font-black tracking-tight ${
                      result.consistency_score >= 50 ? 'text-emerald-600' : 'text-red-600'
                    }`}
                  >
                    {result.consistency_score}%
                  </span>
                  <span className="text-sm text-slate-500 font-medium">agreement across noisy variants</span>
                </div>

                {/* Progress bar visual */}
                <div className="mt-6 pt-4 border-t border-slate-100">
                  <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        result.consistency_score >= 50 ? 'bg-emerald-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(Math.max(result.consistency_score, 0), 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Data Table: Perturbation Analysis */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="px-6 py-5 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <h3 className="text-lg font-bold text-slate-900">Perturbation Analysis</h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Detailed prediction comparison for generated phonetic variations
                  </p>
                </div>
                <span className="inline-flex self-start sm:self-center items-center px-2.5 py-1 rounded-md text-xs font-medium bg-slate-100 text-slate-600">
                  {result.variations ? result.variations.length : 0} Variations Tested
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50/80 border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      <th scope="col" className="px-6 py-3.5">
                        Spelling Variation
                      </th>
                      <th scope="col" className="px-6 py-3.5">
                        Model Prediction
                      </th>
                      <th scope="col" className="px-6 py-3.5 text-right">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-sm">
                    {result.variations && result.variations.length > 0 ? (
                      result.variations.map((v, index) => {
                        const isConsistent = v.pred === result.baseline_pred;
                        return (
                          <tr key={index} className="hover:bg-slate-50/60 transition-colors">
                            <td className="px-6 py-4 font-mono text-slate-800 text-xs sm:text-sm">
                              {v.text}
                            </td>
                            <td className="px-6 py-4">
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
                                {v.pred}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right">
                              {isConsistent ? (
                                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                  <svg className="w-3.5 h-3.5 mr-1 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                  </svg>
                                  Consistent
                                </span>
                              ) : (
                                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-red-50 text-red-700 border border-red-200">
                                  <svg className="w-3.5 h-3.5 mr-1 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                  </svg>
                                  Failed
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={3} className="px-6 py-8 text-center text-slate-400 text-sm">
                          No variations generated.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
