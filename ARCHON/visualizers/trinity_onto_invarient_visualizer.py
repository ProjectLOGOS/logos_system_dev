import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DomainVisualization = () => {
  // Domain data
  const transcendentalDomain = {
    name: "Transcendental Domain",
    values: [
      { name: "EI", value: 1, color: "#8884d8" },
      { name: "OG", value: 2, color: "#82ca9d" },
      { name: "AT", value: 3, color: "#ffc658" }
    ],
    operators: [
      { name: "S₁ᵗ", value: 3, color: "#0088FE" },
      { name: "S₂ᵗ", value: 2, color: "#00C49F" }
    ],
    equation: "1 + 3 - 2 + 2 - 3 = 1",
    invariant: 1,
    invariantName: "Unity Invariant"
  };

  const logicalDomain = {
    name: "Logical Domain",
    values: [
      { name: "ID", value: 1, color: "#8884d8" },
      { name: "NC", value: 2, color: "#82ca9d" },
      { name: "EM", value: 3, color: "#ffc658" }
    ],
    operators: [
      { name: "S₁ᵇ", value: 1, color: "#0088FE" },
      { name: "S₂ᵇ", value: -2, color: "#00C49F" }
    ],
    equation: "1 + 1 + 2 - (-2) - 3 = 3",
    invariant: 3,
    invariantName: "Trinitarian Invariant"
  };

  const mapping = [
    { from: "EI", to: "ID" },
    { from: "OG", to: "NC" },
    { from: "AT", to: "EM" }
  ];

  // Prepare data for visualization
  const getChartData = (domain) => {
    const valueData = domain.values.map(item => ({
      name: item.name,
      value: item.value,
      type: 'Value',
      color: item.color
    }));
    
    const operatorData = domain.operators.map(item => ({
      name: item.name,
      value: item.value,
      type: 'Operator',
      color: item.color
    }));
    
    return [...valueData, ...operatorData];
  };

  // Calculate invariant step by step
  const calculateInvariantSteps = (domain) => {
    if (domain.name === "Transcendental Domain") {
      const v = domain.values;
      const o = domain.operators;
      return [
        { step: 1, expression: `${v[0].name} = ${v[0].value}`, result: v[0].value },
        { step: 2, expression: `${v[0].value} + ${o[0].name} = ${v[0].value} + ${o[0].value} = ${v[0].value + o[0].value}`, result: v[0].value + o[0].value },
        { step: 3, expression: `${v[0].value + o[0].value} - ${v[1].name} = ${v[0].value + o[0].value} - ${v[1].value} = ${v[0].value + o[0].value - v[1].value}`, result: v[0].value + o[0].value - v[1].value },
        { step: 4, expression: `${v[0].value + o[0].value - v[1].value} + ${o[1].name} = ${v[0].value + o[0].value - v[1].value} + ${o[1].value} = ${v[0].value + o[0].value - v[1].value + o[1].value}`, result: v[0].value + o[0].value - v[1].value + o[1].value },
        { step: 5, expression: `${v[0].value + o[0].value - v[1].value + o[1].value} - ${v[2].name} = ${v[0].value + o[0].value - v[1].value + o[1].value} - ${v[2].value} = ${v[0].value + o[0].value - v[1].value + o[1].value - v[2].value}`, result: v[0].value + o[0].value - v[1].value + o[1].value - v[2].value }
      ];
    } else {
      const v = domain.values;
      const o = domain.operators;
      return [
        { step: 1, expression: `${v[0].name} = ${v[0].value}`, result: v[0].value },
        { step: 2, expression: `${v[0].value} + ${o[0].name} = ${v[0].value} + ${o[0].value} = ${v[0].value + o[0].value}`, result: v[0].value + o[0].value },
        { step: 3, expression: `${v[0].value + o[0].value} + ${v[1].name} = ${v[0].value + o[0].value} + ${v[1].value} = ${v[0].value + o[0].value + v[1].value}`, result: v[0].value + o[0].value + v[1].value },
        { step: 4, expression: `${v[0].value + o[0].value + v[1].value} - (${o[1].name}) = ${v[0].value + o[0].value + v[1].value} - (${o[1].value}) = ${v[0].value + o[0].value + v[1].value - o[1].value}`, result: v[0].value + o[0].value + v[1].value - o[1].value },
        { step: 5, expression: `${v[0].value + o[0].value + v[1].value - o[1].value} - ${v[2].name} = ${v[0].value + o[0].value + v[1].value - o[1].value} - ${v[2].value} = ${v[0].value + o[0].value + v[1].value - o[1].value - v[2].value}`, result: v[0].value + o[0].value + v[1].value - o[1].value - v[2].value }
      ];
    }
  };

  const [activeTab, setActiveTab] = useState('transcendental');
  const [showSteps, setShowSteps] = useState(false);

  const transcendentalData = getChartData(transcendentalDomain);
  const logicalData = getChartData(logicalDomain);
  const transcendentalSteps = calculateInvariantSteps(transcendentalDomain);
  const logicalSteps = calculateInvariantSteps(logicalDomain);

  return (
    <div className="flex flex-col p-4 bg-gray-50 rounded-lg shadow-md">
      <h1 className="text-2xl font-bold text-center mb-4">Trinitarian Domain Invariant System</h1>
      
      {/* Tab Selector */}
      <div className="flex mb-4">
        <button
          className={`px-4 py-2 mr-2 rounded-t-lg ${activeTab === 'transcendental' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setActiveTab('transcendental')}
        >
          Transcendental Domain
        </button>
        <button
          className={`px-4 py-2 mr-2 rounded-t-lg ${activeTab === 'logical' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setActiveTab('logical')}
        >
          Logical Domain
        </button>
        <button
          className={`px-4 py-2 rounded-t-lg ${activeTab === 'mapping' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setActiveTab('mapping')}
        >
          Bijective Mapping
        </button>
      </div>
      
      {/* Transcendental Domain View */}
      {activeTab === 'transcendental' && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-2">{transcendentalDomain.name}</h2>
          <p className="mb-4"><strong>Equation:</strong> {transcendentalDomain.equation}</p>
          <p className="mb-4"><strong>Invariant:</strong> {transcendentalDomain.invariant} ({transcendentalDomain.invariantName})</p>
          
          <div className="h-64 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={transcendentalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mb-4"
            onClick={() => setShowSteps(!showSteps)}
          >
            {showSteps ? 'Hide' : 'Show'} Calculation Steps
          </button>
          
          {showSteps && (
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <h3 className="font-semibold mb-2">Calculation Steps:</h3>
              {transcendentalSteps.map((step) => (
                <div key={step.step} className="mb-2 flex">
                  <span className="font-bold mr-2">{step.step}.</span>
                  <span>{step.expression} = <strong>{step.result}</strong></span>
                </div>
              ))}
              <div className="mt-2 p-2 bg-blue-100 rounded">
                <strong>Final Result:</strong> {transcendentalDomain.invariant} ({transcendentalDomain.invariantName})
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Logical Domain View */}
      {activeTab === 'logical' && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-2">{logicalDomain.name}</h2>
          <p className="mb-4"><strong>Equation:</strong> {logicalDomain.equation}</p>
          <p className="mb-4"><strong>Invariant:</strong> {logicalDomain.invariant} ({logicalDomain.invariantName})</p>
          
          <div className="h-64 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={logicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Value" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mb-4"
            onClick={() => setShowSteps(!showSteps)}
          >
            {showSteps ? 'Hide' : 'Show'} Calculation Steps
          </button>
          
          {showSteps && (
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <h3 className="font-semibold mb-2">Calculation Steps:</h3>
              {logicalSteps.map((step) => (
                <div key={step.step} className="mb-2 flex">
                  <span className="font-bold mr-2">{step.step}.</span>
                  <span>{step.expression} = <strong>{step.result}</strong></span>
                </div>
              ))}
              <div className="mt-2 p-2 bg-green-100 rounded">
                <strong>Final Result:</strong> {logicalDomain.invariant} ({logicalDomain.invariantName})
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Bijective Mapping View */}
      {activeTab === 'mapping' && (
        <div className="p-4 bg-white rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Bijective Mapping (λ: T → L)</h2>
          
          <div className="flex justify-center items-center mb-8">
            <div className="text-center p-4 bg-blue-100 rounded-lg mr-8">
              <h3 className="font-semibold mb-2">Transcendental Domain</h3>
              <div className="flex flex-col">
                {transcendentalDomain.values.map((item) => (
                  <div key={item.name} className="mb-2 font-mono">
                    {item.name} = {item.value}
                  </div>
                ))}
              </div>
              <div className="mt-2 p-2 bg-blue-200 rounded">
                Invariant: {transcendentalDomain.invariant}
              </div>
            </div>
            
            <div className="flex flex-col items-center">
              {mapping.map((map, index) => (
                <div key={index} className="mb-2 font-mono">
                  λ({map.from}) = {map.to}
                  <span className="ml-2 text-blue-500">→</span>
                </div>
              ))}
            </div>
            
            <div className="text-center p-4 bg-green-100 rounded-lg ml-8">
              <h3 className="font-semibold mb-2">Logical Domain</h3>
              <div className="flex flex-col">
                {logicalDomain.values.map((item) => (
                  <div key={item.name} className="mb-2 font-mono">
                    {item.name} = {item.value}
                  </div>
                ))}
              </div>
              <div className="mt-2 p-2 bg-green-200 rounded">
                Invariant: {logicalDomain.invariant}
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-yellow-50 rounded border border-yellow-200">
            <h3 className="font-semibold mb-2">Mapping Properties:</h3>
            <p><strong>Structure-Preserving:</strong> Yes</p>
            <p><strong>Function Type:</strong> Bijective (One-to-One and Onto)</p>
            <p><strong>Domain:</strong> T = {'{EI, OG, AT}'}</p>
            <p><strong>Codomain:</strong> L = {'{ID, NC, EM}'}</p>
            <p><strong>Formal Notation:</strong> λ: T<sup>A</sup> → L</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DomainVisualization;