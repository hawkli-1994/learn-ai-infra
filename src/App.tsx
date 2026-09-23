import { Routes, Route } from 'react-router'
import Landing from './pages/Landing'
import Home from './pages/Home'
import GpuLab from './pages/GpuLab'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/flow" element={<Home />} />
      <Route path="/gpu" element={<GpuLab />} />
    </Routes>
  )
}
