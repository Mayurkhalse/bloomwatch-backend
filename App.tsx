
import { useState, useMemo, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navigation/Navbar';
import Sidebar from './components/Navigation/Sidebar';
import BloomMap from './components/Map/BloomMap';
import BloomPopup from './components/Map/BloomPopup';
import Dashboard from './components/Dashboard/Dashboard';
import TimeSlider from './components/Common/TimeSlider';
import LoginForm from './components/Auth/LoginForm';
import SignupForm from './components/Auth/SignupForm';
import { mockBloomEvents, mockRegions, BloomEvent } from './data/mockData';

function App() {
  // Authentication state
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);
  const [showLogin, setShowLogin] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // UI state
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Map and bloom state
  const [selectedBloom, setSelectedBloom] = useState<BloomEvent | null>(null);
  const [isBloomPopupOpen, setIsBloomPopupOpen] = useState(false);
  const [blooms, setBlooms] = useState(mockBloomEvents);
  const [selectedRegions, setSelectedRegions] = useState<string[]>(['1']);

  // Region selection state
  const [userRegion, setUserRegion] = useState<{
    bounds: [[number, number], [number, number]] | null;
    isDrawing: boolean;
    isEditing: boolean;
  }>({
    bounds: null,
    isDrawing: false,
    isEditing: false,
  });

  // Time slider state
  const initialDate = useMemo(() => {
    const dates = mockBloomEvents.map(b => b.date).sort();
    return dates[0] ?? '2024-01-01';
  }, []);
  const [currentDate, setCurrentDate] = useState(initialDate);
  const availableDates = useMemo(() => {
    const unique = Array.from(new Set(blooms.map(b => b.date)));
    unique.sort();
    return unique;
  }, [blooms]);

  // Authentication handlers
  const handleLogin = async (email: string, password: string) => {
    try {
      const formData = new FormData();
      formData.append("email", email);
      formData.append("password", password);

      const response = await fetch("http://localhost:8000/login", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUser({ name: data.profile?.username || "User", email });
        setIsAuthenticated(true);

        localStorage.setItem("uid", data.uid || "");
        localStorage.setItem("name", data.profile?.username || "");
        localStorage.setItem("email", email);
        localStorage.setItem("token", data.token || "");
      } else {
        alert(data.error || "Login failed");
      }
    } catch (err) {
      console.error("Login error:", err);
      alert("An error occurred while logging in.");
    }
  };

  const handleSignup = async (name: string, email: string, password: string) => {
    try {
      const formData = new FormData();
      formData.append("username", name);
      formData.append("email", email);
      formData.append("password", password);

      const response = await fetch("http://localhost:8000/addUser", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUser({ name, email });
        setIsAuthenticated(true);

        localStorage.setItem("uid", data.uid || "");
      } else {
        alert(data.error || "Signup failed");
      }
    } catch (err) {
      console.error("Signup error:", err);
      alert("An error occurred while signing up.");
    }
  };

  const handleLogout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.clear();
  };

  // Bloom handlers
  const handleShowBloomDetails = (bloom: BloomEvent) => {
    setSelectedBloom(bloom);
    setIsBloomPopupOpen(true);
  };

  const handleRegionToggle = (regionId: string) => {
    setSelectedRegions(prev =>
      prev.includes(regionId)
        ? prev.filter(id => id !== regionId)
        : [...prev, regionId]
    );
  };

  // Region drawing handlers
  const handleStartDrawing = () => {
    setUserRegion(prev => ({
      ...prev,
      isDrawing: true,
      isEditing: false,
    }));
  };

  const handleStopDrawing = () => {
    setUserRegion(prev => ({
      ...prev,
      isDrawing: false,
    }));
  };

  const handleRegionDrawn = (bounds: [[number, number], [number, number]]) => {
    setUserRegion(prev => ({
      ...prev,
      bounds,
      isDrawing: false,
    }));
  };

  const handleEditRegion = () => {
    setUserRegion(prev => ({
      ...prev,
      isEditing: !prev.isEditing,
    }));
  };

  const handleClearRegion = () => {
    setUserRegion({
      bounds: null,
      isDrawing: false,
      isEditing: false,
    });
  };

  const handleSaveRegion = async () => {
    if (!userRegion.bounds) return;

    const uid = localStorage.getItem("uid");
    if (!uid) {
      alert("User not logged in");
      return;
    }

    const [[lat1, lng1], [lat2, lng2]] = userRegion.bounds;

    const formData = new FormData();
    formData.append("uid", uid);
    formData.append("lat_1", lat1.toString());
    formData.append("lat_2", lat2.toString());
    formData.append("lan_1", lng1.toString());
    formData.append("lan_2", lng2.toString());

    try {
      const response = await fetch("http://localhost:8000/addRegion", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        alert(data.message || "Region added successfully!");
      } else {
        alert(data.error || "Failed to add region");
      }
    } catch (err) {
      console.error("Add region error:", err);
      alert("An error occurred while adding the region");
    }
  };

  // Filter blooms by current date and userRegion
  const bloomsOnDate = useMemo(
    () => blooms.filter(b => b.date === currentDate),
    [blooms, currentDate]
  );

  const filteredBlooms = userRegion.bounds
    ? bloomsOnDate.filter(bloom => {
        const [lat, lng] = bloom.coordinates;
        const [[minLat, minLng], [maxLat, maxLng]] = userRegion.bounds!;
        return lat >= minLat && lat <= maxLat && lng >= minLng && lng <= maxLng;
      })
    : [];

  // Get UID from localStorage
  const uid = localStorage.getItem("uid") || '';

  // Show auth forms if not logged in
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center">
        {showLogin ? (
          <LoginForm
            onLogin={handleLogin}
            onToggleForm={() => setShowLogin(false)}
          />
        ) : (
          <SignupForm
            onSignup={handleSignup}
            onToggleForm={() => setShowLogin(true)}
          />
        )}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      <Navbar
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onLogout={handleLogout}
        user={user}
      />

      <Sidebar isOpen={isSidebarOpen} />

      <main className={`transition-all duration-300 ${isSidebarOpen ? 'ml-64' : 'ml-0'} mt-16`}>
        <Routes>
          <Route path="/" element={<Navigate to="/map" replace />} />
          <Route
            path="/map"
            element={
              <div className="h-screen flex flex-col">
                <div className="flex-1 relative">
                  <BloomMap
                    blooms={filteredBlooms}
                    onShowBloomDetails={handleShowBloomDetails}
                    userRegion={userRegion}
                    onRegionDrawn={handleRegionDrawn}
                    onStopDrawing={handleStopDrawing}
                    uid={uid} // Pass UID for fetching saved region
                  />
                  {/* Region control panel */}
                  <div className="absolute top-4 right-4 z-20">
                    <div className="bg-gray-800/90 backdrop-blur-md p-3 rounded-lg shadow-lg">
                      <div className="flex space-x-2">
                        <button
                          onClick={handleStartDrawing}
                          className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                            userRegion.isDrawing
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                          }`}
                        >
                          {userRegion.isDrawing ? 'Drawing...' : 'Draw Region'}
                        </button>
                        {userRegion.isDrawing && (
                          <button
                            onClick={handleStopDrawing}
                            className="px-3 py-2 text-sm rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors"
                          >
                            Cancel
                          </button>
                        )}
                        {userRegion.bounds && !userRegion.isDrawing && (
                          <>
                            <button
                              onClick={handleEditRegion}
                              className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                                userRegion.isEditing
                                  ? 'bg-yellow-600 text-white'
                                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                              }`}
                            >
                              {userRegion.isEditing ? 'Editing...' : 'Edit'}
                            </button>
                            <button
                              onClick={handleClearRegion}
                              className="px-3 py-2 text-sm rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors"
                            >
                              Clear
                            </button>
                            <button
                              onClick={handleSaveRegion}
                              className="px-3 py-2 text-sm rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                            >
                              Save Region
                            </button>
                          </>
                        )}
                      </div>
                      {userRegion.isDrawing && (
                        <p className="text-xs text-blue-400 mt-2">
                          Click and drag to draw a rectangle on the map<br />
                          Press Escape to cancel
                        </p>
                      )}
                      {!userRegion.bounds && !userRegion.isDrawing && (
                        <p className="text-xs text-gray-400 mt-2">
                          Draw a region to view bloom data
                        </p>
                      )}
                      {userRegion.bounds && !userRegion.isDrawing && (
                        <div className="mt-2 space-y-1">
                          <p className="text-xs text-green-400">
                            Region active - {filteredBlooms.length} blooms visible
                          </p>
                          <p className="text-xs text-gray-500">
                            Bounds: {userRegion.bounds[0][0].toFixed(2)}, {userRegion.bounds[0][1].toFixed(2)} to {userRegion.bounds[1][0].toFixed(2)}, {userRegion.bounds[1][1].toFixed(2)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20">
                    <TimeSlider
                      dates={availableDates}
                      currentDate={currentDate}
                      onDateChange={setCurrentDate}
                    />
                  </div>
                </div>
              </div>
            }
          />
          <Route
            path="/dashboard"
            element={
              <Dashboard
                blooms={filteredBlooms}
                regions={mockRegions}
                selectedRegions={selectedRegions}
                onRegionToggle={handleRegionToggle}
              />
            }
          />
          <Route path="*" element={<Navigate to="/map" replace />} />
        </Routes>
      </main>

      <BloomPopup
        bloom={selectedBloom}
        isOpen={isBloomPopupOpen}
        onClose={() => {
          setIsBloomPopupOpen(false);
          setSelectedBloom(null);
        }}
      />
    </div>
  );
}

export default App;