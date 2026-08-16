import React, { useState, useEffect, useMemo } from 'react';
import { 
  BookOpen, Sparkles, Compass, Search, Star, Globe, 
  RotateCw, CheckCircle2, Sliders, History, Info, ChevronRight, Layers
} from 'lucide-react';
import confetti from 'canvas-confetti';

import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const API_BASE = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' ? "http://localhost:8000/api" : "/api");

const POPULAR_AFRICAN_COUNTRIES = [
  "Nigeria", "South Africa", "Kenya", "Ghana", "Egypt", 
  "Ethiopia", "Uganda", "Tanzania", "Rwanda", "Morocco"
];

export default function App() {
  const [countries, setCountries] = useState([]);
  const [session, setSession] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('discover'); // 'discover' | 'search' | 'radar' | 'history'
  
  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Modal / Toast
  const [showOnboardModal, setShowOnboardModal] = useState(false);
  const [selectedCountry, setSelectedCountry] = useState('Nigeria');
  const [toast, setToast] = useState(null);

  // 1. Fetch Countries on Mount
  useEffect(() => {
    fetch(`${API_BASE}/countries`)
      .then(res => res.json())
      .then(data => {
        if (data.countries) {
          setCountries(data.countries);
        }
      })
      .catch(err => console.error("Error fetching countries:", err));
  }, []);

  // 2. Auto-onboard default country (Nigeria) if no session
  useEffect(() => {
    if (!session && countries.length > 0) {
      handleOnboard('Nigeria');
    }
  }, [countries]);

  const showToast = (message) => {
    setToast(message);
    setTimeout(() => setToast(null), 4000);
  };

  const handleOnboard = async (countryName) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/onboard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ country: countryName, language: 'eng' })
      });
      const data = await res.json();
      setSession({
        user_id: data.user_id,
        country: data.country,
        cultural_profile: data.cultural_profile,
        history: []
      });
      setRecommendations(data.recommendations || []);
      setShowOnboardModal(false);
      showToast(`Welcome! Initialized cultural profile for ${countryName}.`);
    } catch (err) {
      console.error("Onboarding error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRateBook = async (bookId, ratingScore) => {
    if (!session) return;
    try {
      const res = await fetch(`${API_BASE}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: session.user_id,
          book_id: bookId,
          rating: ratingScore
        })
      });
      const data = await res.json();
      if (data.status === 'success') {
        // Trigger celebratory confetti
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.85 }
        });

        // Update session state
        setSession(prev => ({
          ...prev,
          cultural_profile: data.cultural_profile,
          history: [
            { book_id: bookId, rating: ratingScore, time: new Date().toLocaleTimeString() },
            ...(prev?.history || [])
          ]
        }));

        // Dynamic re-recommendation
        setRecommendations(data.recommendations);
        showToast(`Rated ${ratingScore}★! Your Hofstede cultural vector has updated.`);
      }
    } catch (err) {
      console.error("Rating error:", err);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const uParam = session ? `&user_id=${session.user_id}` : '';
      const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(searchQuery)}${uParam}`);
      const data = await res.json();
      setSearchResults(data.results || []);
    } catch (err) {
      console.error("Search error:", err);
    } finally {
      setIsSearching(false);
    }
  };

  // Radar Chart Data Preparation
  const radarChartData = useMemo(() => {
    if (!session?.cultural_profile) return null;
    const p = session.cultural_profile;
    return {
      labels: [
        'Power Distance (PDI)',
        'Individualism (IDV)',
        'Masculinity (MAS)',
        'Uncertainty Avoidance (UAI)',
        'Long-Term Orientation (LTO)',
        'Indulgence (IVR)'
      ],
      datasets: [
        {
          label: `${session.country} Inferred Profile`,
          data: [p.pdi, p.idv, p.mas, p.uai, p.lto, p.ivr],
          backgroundColor: 'rgba(245, 158, 11, 0.25)',
          borderColor: '#f59e0b',
          borderWidth: 2,
          pointBackgroundColor: '#f59e0b',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#f59e0b',
        },
        {
          label: 'Global Average Baseline',
          data: [50, 50, 50, 50, 50, 50],
          backgroundColor: 'rgba(6, 182, 212, 0.08)',
          borderColor: 'rgba(6, 182, 212, 0.5)',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
        }
      ]
    };
  }, [session]);

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toast && (
        <div className="toast-container">
          <CheckCircle2 size={20} color="#10b981" />
          <span>{toast}</span>
        </div>
      )}

      {/* Header Bar */}
      <header className="app-header">
        <div className="logo-container" onClick={() => setActiveTab('discover')}>
          <div className="logo-icon">
            <BookOpen size={20} color="#090d16" />
          </div>
          <div>
            <h1 className="brand-title" style={{ fontSize: '1.2rem', fontWeight: 800 }}>
              AfriRead <span style={{ color: 'var(--accent-gold)' }}>AI</span>
            </h1>
            <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              Culturally Aware Recommender
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button 
            className={`nav-btn ${activeTab === 'discover' ? 'active' : ''}`}
            onClick={() => setActiveTab('discover')}
          >
            <Sparkles size={15} /> Discover
          </button>
          <button 
            className={`nav-btn ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <Search size={15} /> Search
          </button>
          <button 
            className={`nav-btn ${activeTab === 'radar' ? 'active' : ''}`}
            onClick={() => setActiveTab('radar')}
          >
            <Compass size={15} /> Radar
          </button>
          <button 
            className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={15} /> History ({session?.history?.length || 0})
          </button>
        </nav>

        {/* User Cultural Country Pill */}
        <div 
          className="user-status-pill cursor-pointer"
          style={{ cursor: 'pointer' }}
          onClick={() => setShowOnboardModal(true)}
          title="Click to switch cultural origin"
        >
          <Globe size={15} color="var(--accent-cyan)" />
          <span>
            <strong>{session?.country || 'Select'}</strong>
          </span>
          <span style={{ color: 'var(--accent-gold)', fontSize: '0.75rem', marginLeft: '0.15rem' }}>
            ▾
          </span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="hero-section">
        {/* Tab 1: Discover / Recommended Shelf */}
        {activeTab === 'discover' && (
          <div>
            <div className="hero-banner glass-panel">
              <div>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.25rem 0.75rem', borderRadius: '9999px', background: 'rgba(245, 158, 11, 0.15)', color: 'var(--accent-gold)', fontSize: '0.78rem', fontWeight: 700, marginBottom: '0.65rem' }}>
                  <Sparkles size={14} /> Hybrid AI Engine (FM v2 + SVD++)
                </div>
                <h2 className="hero-title">
                  Recommended For You in <span style={{ color: 'var(--accent-gold)' }}>{session?.country}</span>
                </h2>
                <p className="hero-desc">
                  Recommendations are dynamically calibrated to your region's Hofstede cultural dimensions. Rate books below to refine your cultural taste profile in real-time.
                </p>
              </div>

              <button 
                onClick={() => handleOnboard(session?.country || 'Nigeria')}
                className="nav-btn active"
                style={{ padding: '0.75rem 1.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}
              >
                <RotateCw size={15} className={loading ? "animate-spin" : ""} /> Refresh Shelf
              </button>
            </div>

            {/* Recommendations Grid */}
            <div className="book-grid">
              {recommendations.map((b) => (
                <BookCard 
                  key={b.book_id} 
                  book={b} 
                  onRate={handleRateBook}
                />
              ))}
            </div>
          </div>
        )}

        {/* Tab 2: Search Catalog */}
        {activeTab === 'search' && (
          <div>
            <div className="search-card glass-panel">
              <h2 style={{ fontSize: '1.4rem', fontWeight: 800, marginBottom: '0.5rem' }}>
                Search 15,000+ Book Catalog
              </h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
                Search by book title, author, or genre. Every search result displays your real-time predicted rating and cultural affinity match.
              </p>

              <form onSubmit={handleSearch} className="search-form-row">
                <div style={{ flex: 1, position: 'relative' }}>
                  <Search size={18} color="var(--text-muted)" style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)' }} />
                  <input
                    type="text"
                    placeholder="Search titles (e.g. 'Things Fall Apart', 'Fantasy', 'Achebe')..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.8rem 1rem 0.8rem 2.6rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'rgba(15, 23, 42, 0.8)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-primary)',
                      fontSize: '0.9rem',
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
                <button
                  type="submit"
                  className="nav-btn active"
                  style={{ padding: '0.8rem 1.6rem', fontSize: '0.9rem', flexShrink: 0 }}
                >
                  {isSearching ? 'Searching...' : 'Search'}
                </button>
              </form>
            </div>

            {/* Search Results Grid */}
            <div className="book-grid">
              {searchResults.map((b) => (
                <BookCard 
                  key={b.book_id} 
                  book={b} 
                  onRate={handleRateBook}
                />
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Cultural Radar & Hofstede Profile */}
        {activeTab === 'radar' && (
          <div className="radar-grid">
            {/* Left: Radar Chart */}
            <div className="radar-chart-card glass-panel">
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.4rem', alignSelf: 'flex-start' }}>
                Dynamic Cultural Radar
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', alignSelf: 'flex-start' }}>
                Visualizing your 6 Hofstede cultural dimension scores (0–100). As you rate books, this profile evolves.
              </p>

              <div className="radar-chart-wrapper">
                {radarChartData && (
                  <Radar 
                    data={radarChartData} 
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        r: {
                          angleLines: { color: 'rgba(148, 163, 184, 0.15)' },
                          grid: { color: 'rgba(148, 163, 184, 0.15)' },
                          pointLabels: { color: '#94a3b8', font: { size: 10, weight: 600 } },
                          ticks: { display: false, stepSize: 20 },
                          min: 0,
                          max: 100
                        }
                      },
                      plugins: {
                        legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { size: 11 } } }
                      }
                    }}
                  />
                )}
              </div>
            </div>

            {/* Right: Dimension Deep Dive Breakdown */}
            <div className="radar-breakdown-card glass-panel">
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1rem' }}>
                Hofstede Dimensions Breakdown
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
                {session?.cultural_profile && Object.entries(session.cultural_profile.labels || {}).map(([key, label]) => {
                  const val = session.cultural_profile[key] || 50;
                  return (
                    <div key={key} style={{ padding: '0.8rem 1rem', borderRadius: 'var(--radius-md)', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-color)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                        <span style={{ fontSize: '0.825rem', fontWeight: 600 }}>{label}</span>
                        <strong style={{ color: 'var(--accent-gold)', fontSize: '0.875rem' }}>{val}/100</strong>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#334155', borderRadius: '9999px', overflow: 'hidden' }}>
                        <div style={{ width: `${val}%`, height: '100%', background: 'linear-gradient(90deg, var(--accent-gold), var(--accent-cyan))', transition: 'width 0.4s ease' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: User Rated History */}
        {activeTab === 'history' && (
          <div className="glass-panel" style={{ padding: '1.75rem', borderRadius: 'var(--radius-lg)' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.4rem' }}>
              Your Rated Books History
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              Every rating recalibrates your hybrid model weights in real time.
            </p>

            {(!session?.history || session.history.length === 0) ? (
              <div style={{ padding: '2.5rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <BookOpen size={44} style={{ margin: '0 auto 1rem', opacity: 0.4 }} />
                <p style={{ fontSize: '0.9rem' }}>You have not rated any books yet.</p>
                <button 
                  onClick={() => setActiveTab('discover')} 
                  className="nav-btn active"
                  style={{ marginTop: '1rem', display: 'inline-flex' }}
                >
                  Go Discover & Rate Books
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {session.history.map((h, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.85rem 1rem', borderRadius: 'var(--radius-md)', background: 'rgba(15, 23, 42, 0.7)', border: '1px solid var(--border-color)', gap: '0.75rem' }}>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{h.title || `Book ID #${h.book_id}`}</div>
                      <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Rated at {h.time}</div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: 'var(--accent-gold)', fontWeight: 700, fontSize: '0.9rem', flexShrink: 0 }}>
                      <Star size={15} fill="var(--accent-gold)" /> {h.rating}★
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Onboarding Modal */}
      {showOnboardModal && (
        <div className="modal-backdrop" onClick={() => setShowOnboardModal(false)}>
          <div className="modal-card glass-panel" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.5rem' }}>
              Select Cultural Origin
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              The recommendation engine uses Hofstede's 6 cultural dimensions for your country to customize cold-start recommendations.
            </p>

            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
                Popular African Regions
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {POPULAR_AFRICAN_COUNTRIES.map(c => (
                  <button
                    key={c}
                    onClick={() => handleOnboard(c)}
                    className={`genre-tag cursor-pointer ${session?.country === c ? 'active' : ''}`}
                    style={{
                      padding: '0.4rem 0.8rem',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                      border: session?.country === c ? '1px solid var(--accent-gold)' : '1px solid var(--border-color)',
                      background: session?.country === c ? 'rgba(245, 158, 11, 0.2)' : 'rgba(30, 41, 59, 0.6)',
                      color: session?.country === c ? 'var(--accent-gold)' : 'var(--text-primary)'
                    }}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: '2rem' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block', marginBottom: '0.5rem' }}>
                All 119 Hofstede Countries
              </label>
              <select
                value={selectedCountry}
                onChange={(e) => setSelectedCountry(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem 1rem',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                  outline: 'none'
                }}
              >
                {countries.map(c => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button 
                onClick={() => setShowOnboardModal(false)}
                className="nav-btn"
              >
                Cancel
              </button>
              <button 
                onClick={() => handleOnboard(selectedCountry)}
                className="nav-btn active"
                style={{ padding: '0.6rem 1.5rem' }}
              >
                Confirm Country
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sub-Component: Book Card ────────────────────────────────────────────────
function BookCard({ book, onRate }) {
  const [userRating, setUserRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [imgError, setImgError] = useState(false);

  const coverUrl = useMemo(() => {
    if (imgError || !book.image_url || book.image_url.includes('nophoto')) {
      return null;
    }
    return book.image_url;
  }, [book, imgError]);

  return (
    <div className="book-card glass-panel">
      {/* Cover Area */}
      <div className="book-cover-container">
        {coverUrl ? (
          <img 
            src={coverUrl} 
            alt={book.title} 
            className="book-cover-img"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="cover-placeholder">
            <BookOpen size={36} color="var(--accent-gold)" style={{ opacity: 0.6, marginBottom: '0.5rem' }} />
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
              {book.title}
            </div>
          </div>
        )}

        {/* Badges */}
        <div className="book-badge-cultural">
          <Globe size={12} />
          {book.cultural_alignment}% Match
        </div>
        <div className="book-badge-rating">
          <Star size={12} fill="var(--accent-gold)" />
          {book.predicted_rating}★
        </div>
      </div>

      {/* Book Metadata Content */}
      <div className="book-content">
        <h4 className="book-title" title={book.title}>
          {book.title}
        </h4>
        <p className="book-author">
          by {Array.isArray(book.authors) ? book.authors.join(", ") : (book.authors || "Unknown")}
        </p>

        {/* Genres & Tags */}
        <div className="genre-pills">
          {(book.genres && book.genres.length > 0) ? (
            book.genres.map((g, idx) => (
              <span key={idx} className="genre-tag">
                {g}
              </span>
            ))
          ) : (
            <span className="genre-tag">General Literature</span>
          )}
        </div>

        {/* Book Description Snippet */}
        <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: '1.4', margin: '0.25rem 0', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
          {book.description || "A celebrated literary title curated for your regional cultural preferences."}
        </p>

        {/* Interactive Rating Row */}
        <div className="rating-action-row">
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            Your Rating:
          </span>
          <div className="star-group">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                className={`star-btn ${star <= (hoverRating || userRating) ? 'active' : ''}`}
                onClick={() => {
                  setUserRating(star);
                  onRate(book.book_id, star);
                }}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(0)}
                title={`Rate ${star} Stars`}
              >
                <Star size={16} fill={star <= (hoverRating || userRating) ? "var(--accent-gold)" : "none"} />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
