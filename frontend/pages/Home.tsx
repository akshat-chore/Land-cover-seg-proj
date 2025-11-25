import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Map, Droplets, ArrowRight } from 'lucide-react';

const Home: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-mntn-dark relative overflow-hidden flex items-center justify-center">
      
      {/* Background Image with Overlay */}
      <div 
        className="absolute inset-0 z-0 bg-cover bg-center opacity-30 scale-105"
        style={{ backgroundImage: 'url("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?ixlib=rb-1.2.1&auto=format&fit=crop&w=2000&q=80")' }}
      ></div>
      <div className="absolute inset-0 z-0 bg-gradient-to-t from-mntn-dark via-mntn-dark/90 to-transparent"></div>

      <div className="relative z-10 max-w-7xl w-full px-6 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
        
        {/* Left: Text Content */}
        <div className="text-white space-y-6 animate-in slide-in-from-left duration-1000">
            <h2 className="text-mntn-accent uppercase tracking-[0.2em] font-medium text-xs md:text-sm">
                Landcover-Driven Intelligence
            </h2>
            <h1 className="font-serif text-4xl md:text-6xl font-bold leading-tight">
                Smart Urban <br/> Decision Making
            </h1>
            <p className="text-gray-300 text-lg md:text-xl font-light max-w-lg leading-relaxed border-l-2 border-mntn-accent/50 pl-6">
                Landcover-Driven Flood Simulation and Infrastructure Equity Analysis. 
                <br/><br/>
                <span className="text-sm text-gray-400">
                    Leveraging advanced computer vision to decode satellite imagery. We provide actionable intelligence on infrastructure deficits and flood risks to build resilient, equitable cities.
                </span>
            </p>
        </div>

        {/* Right: Modules */}
        <div className="grid gap-6 animate-in slide-in-from-bottom duration-1000 delay-200">
            
            {/* Infrastructure Equity Module */}
            <div onClick={() => navigate('/analyze/deficit')} className="group cursor-pointer">
                <div className="bg-white/5 backdrop-blur-md border border-white/10 p-8 rounded-2xl hover:bg-white/10 transition-all duration-300 hover:scale-[1.02] hover:border-mntn-accent/50 shadow-lg">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-4 bg-orange-500/20 rounded-xl text-orange-400 border border-orange-500/20">
                            <Map className="w-8 h-8" />
                        </div>
                        <ArrowRight className="w-6 h-6 text-gray-500 group-hover:text-white transition-colors transform group-hover:translate-x-1" />
                    </div>
                    <h3 className="text-2xl font-serif text-white mb-2">Infrastructure Equity Analysis</h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                        Detect, map, and quantify infrastructure gaps. Analyze land-cover patterns to identify deficits in road coverage, green spaces, and building density in underserved regions.
                    </p>
                </div>
            </div>

            {/* Flood Simulation Module */}
            <div onClick={() => navigate('/analyze/flood')} className="group cursor-pointer">
                <div className="bg-white/5 backdrop-blur-md border border-white/10 p-8 rounded-2xl hover:bg-white/10 transition-all duration-300 hover:scale-[1.02] hover:border-mntn-accent/50 shadow-lg">
                    <div className="flex justify-between items-start mb-4">
                        <div className="p-4 bg-blue-500/20 rounded-xl text-blue-400 border border-blue-500/20">
                            <Droplets className="w-8 h-8" />
                        </div>
                        <ArrowRight className="w-6 h-6 text-gray-500 group-hover:text-white transition-colors transform group-hover:translate-x-1" />
                    </div>
                    <h3 className="text-2xl font-serif text-white mb-2">Flood Risk Simulation</h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                        Predict disaster impact before it happens. Simulate rising water levels over current land cover to identify critical building vulnerabilities and assess urban resilience.
                    </p>
                </div>
            </div>

        </div>
      </div>

      <div className="absolute bottom-6 left-0 right-0 text-center">
        <p className="text-gray-600 text-[10px] tracking-[0.3em] uppercase opacity-60">
            MNTN Intelligence System • v2.0
        </p>
      </div>

    </div>
  );
};

export default Home;