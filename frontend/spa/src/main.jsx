import React from 'react'
import { createRoot } from 'react-dom/client'
import Portfolio from './Portfolio'
import './styles.css'

function App() {
  const [route, setRoute] = React.useState('portfolio')

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: 24 }}>
      <header style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 18 }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>AlgoTrendy SPA</h1>
        <nav>
          <button onClick={() => setRoute('portfolio')} style={{ marginRight: 8 }} aria-current={route === 'portfolio'}>Portfolio</button>
        </nav>
      </header>
      <main>
        {route === 'portfolio' && <Portfolio />}
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
