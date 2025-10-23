import { useState, useEffect } from 'react';
import Viewer3D from './components/Viewer3D';
import ResultsList from './components/ResultsList';
import ControlPanel from './components/ControlPanel';
import FileDownload from './components/FileDownload';
import { useResults, useResult } from './hooks/useResults';
import { usePointCloud } from './hooks/usePointCloud';
import './App.css';

function App() {
  const { results, loading: resultsLoading } = useResults();
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);
  const { result } = useResult(selectedResultId);
  
  const [showInput, setShowInput] = useState(true);
  const [showSample, setShowSample] = useState(true);
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);

  // Load point clouds
  const { geometry: inputGeometry } = usePointCloud(selectedResultId, 'input', 0);
  const { geometry: sampleGeometry } = usePointCloud(selectedResultId, 'sample', currentSampleIndex);

  // Get point counts
  const inputPoints = inputGeometry?.attributes.position?.count || 0;
  const samplePoints = sampleGeometry?.attributes.position?.count || 0;

  // Auto-select first result if none selected
  useEffect(() => {
    if (!selectedResultId && results.length > 0) {
      setSelectedResultId(results[0].id);
    }
  }, [results, selectedResultId]);

  // Reset sample index when result changes
  useEffect(() => {
    setCurrentSampleIndex(0);
  }, [selectedResultId]);

  const handleToggleInput = () => setShowInput(!showInput);
  const handleToggleSample = () => setShowSample(!showSample);
  const handleShowBoth = () => {
    setShowInput(true);
    setShowSample(true);
  };
  const handleResetCamera = () => {
    // Camera reset is handled by OrbitControls in Viewer3D
    console.log('Reset camera');
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>PCDiff Web Viewer</h1>
        <p>Interactive 3D visualization of skull implant generation results</p>
      </header>

      <div className="app-layout">
        {/* Left Sidebar - Results List */}
        <aside className="sidebar sidebar-left">
          <ResultsList
            results={results}
            selectedResultId={selectedResultId}
            onSelect={setSelectedResultId}
            loading={resultsLoading}
          />
        </aside>

        {/* Main Content - 3D Viewer */}
        <main className="main-content">
          {selectedResultId ? (
            <Viewer3D
              inputGeometry={inputGeometry}
              sampleGeometry={sampleGeometry}
              showInput={showInput}
              showSample={showSample}
              onCameraReset={handleResetCamera}
            />
          ) : (
            <div className="empty-state">
              <h2>No Result Selected</h2>
              <p>Select an inference result from the left panel to view</p>
            </div>
          )}
        </main>

        {/* Right Sidebar - Controls & Downloads */}
        <aside className="sidebar sidebar-right">
          <div className="sidebar-section">
            <ControlPanel
              showInput={showInput}
              showSample={showSample}
              onToggleInput={handleToggleInput}
              onToggleSample={handleToggleSample}
              onShowBoth={handleShowBoth}
              onResetCamera={handleResetCamera}
              numSamples={result?.num_samples || 1}
              currentSampleIndex={currentSampleIndex}
              onSampleChange={setCurrentSampleIndex}
              inputPoints={inputPoints}
              samplePoints={samplePoints}
            />
          </div>
          <div className="sidebar-section">
            <FileDownload resultId={selectedResultId} />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;

