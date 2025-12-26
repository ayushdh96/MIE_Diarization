import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { Button } from "@/components/ui/button"
import './App.css'
import MicRecorderComponent from './components/ui/MicRecorder'
import { Github, Linkedin } from 'lucide-react'

function App() {
  const [count, setCount] = useState(0)
  const [backendMessage, setBackendMessage] = useState('');

  useEffect(() => {
    fetch('http://127.0.0.1:5001/api/test')
      .then((res) => res.json())
      .then((data) => {setBackendMessage(data.message)
        console.log('Backend message:', data.message);
      })
      .catch((err) => console.error('Error fetching backend:', err));
  }, []);

  return (
    <>
      <div className="min-h-screen flex items-center justify-center bg-gray-100 relative">
        {/* Social Links - Top Right Corner */}
        <div className="absolute top-6 right-6 flex gap-3 z-10">
          <a
            href="https://github.com/ayushdh96"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-900 transition-colors"
            title="GitHub"
          >
            <Github className="w-8 h-8" />
          </a>
          <a
            href="https://www.linkedin.com/in/ayush-dhoundiyal-b92374191"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-blue-600 transition-colors"
            title="LinkedIn"
          >
            <Linkedin className="w-8 h-8" />
          </a>
        </div>
        
        <MicRecorderComponent />
      </div>
    </>
  )
}

export default App
