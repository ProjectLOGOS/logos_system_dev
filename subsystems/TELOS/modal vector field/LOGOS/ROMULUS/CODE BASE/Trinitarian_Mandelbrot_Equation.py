/**
 * Trinitarian Mandelbrot Equation derived from theological mathematics
 * 
 * This implementation represents the mathematical concepts found in the theological 
 * framework that incorporates the Trinity structure, resurrection mechanics, and 
 * ontological transformations using complex numbers.
 */

class Complex {
  constructor(real, imag) {
    this.real = real;
    this.imag = imag;
  }
  
  add(other) {
    if (typeof other === 'number') {
      return new Complex(this.real + other, this.imag);
    }
    return new Complex(this.real + other.real, this.imag + other.imag);
  }
  
  multiply(other) {
    if (typeof other === 'number') {
      return new Complex(this.real * other, this.imag * other);
    }
    
    const real = this.real * other.real - this.imag * other.imag;
    const imag = this.real * other.imag + this.imag * other.real;
    return new Complex(real, imag);
  }
  
  pow(n) {
    if (n === 0) return new Complex(1, 0);
    if (n === 1) return this;
    
    let result = this;
    for (let i = 1; i < n; i++) {
      result = result.multiply(this);
    }
    return result;
  }
  
  divide(other) {
    if (typeof other === 'number') {
      if (other === 0) throw new Error("Division by zero");
      return new Complex(this.real / other, this.imag / other);
    }
    
    const denominator = other.real * other.real + other.imag * other.imag;
    if (denominator === 0) throw new Error("Division by zero");
    
    const real = (this.real * other.real + this.imag * other.imag) / denominator;
    const imag = (this.imag * other.real - this.real * other.imag) / denominator;
    
    return new Complex(real, imag);
  }
  
  abs() {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }
  
  toString() {
    if (this.imag === 0) return this.real.toString();
    if (this.real === 0) return this.imag === 1 ? "i" : this.imag === -1 ? "-i" : this.imag + "i";
    return `${this.real}${this.imag >= 0 ? "+" : ""}${this.imag === 1 ? "i" : this.imag === -1 ? "-i" : this.imag + "i"}`;
  }
}

/**
 * Calculates the Trinitarian Mandelbrot iteration for a given point c
 * 
 * The equation: z_{n+1} = (z_n³ + z_n² + z_n + c) / (i^(|z_n| mod 4) + 1)
 * 
 * Where:
 * - z_n³ represents the Spirit (3)
 * - z_n² represents the Son (2)
 * - z_n represents the Father (1)
 * - i^(|z_n| mod 4) cycles through ontological states:
 *   * i^0 = 1: Identity/Being (Father)
 *   * i^1 = i: Transition
 *   * i^2 = -1: Death/Negation
 *   * i^3 = -i: Return/Resurrection
 * - Division represents the unity-in-trinity relationship
 * 
 * @param {Complex|number} c - The parameter c in the Mandelbrot equation
 * @param {number} maxIterations - Maximum iterations to perform
 * @param {number} escapeRadius - Radius at which a point is considered escaped
 * @returns {Object} Result of the iteration including iteration count and path
 */
function trinitarianMandelbrot(c, maxIterations = 100, escapeRadius = 2) {
  // Convert number parameter to complex
  if (typeof c === 'number') {
    c = new Complex(c, 0);
  }
  
  // Initialize z to 0 (representing God/Unity)
  let z = new Complex(0, 0);
  let path = [];
  
  for (let iter = 0; iter < maxIterations; iter++) {
    // Calculate the powers of z representing the Trinity
    const z3 = z.pow(3); // Spirit (3)
    const z2 = z.pow(2); // Son (2)
    // z is already Father (1)
    
    // Calculate numerator: z³ + z² + z + c
    const numerator = z3.add(z2).add(z).add(c);
    
    // Calculate i^(|z|) which cycles through theological states
    const zMagnitude = z.abs();
    const iPower = Math.floor(zMagnitude) % 4;
    
    let iPowResult;
    if (iPower === 0) {
      iPowResult = new Complex(1, 0); // i^0 = 1 (Father, identity)
    } else if (iPower === 1) {
      iPowResult = new Complex(0, 1); // i^1 = i (transition)
    } else if (iPower === 2) {
      iPowResult = new Complex(-1, 0); // i^2 = -1 (Death)
    } else { // iPower === 3
      iPowResult = new Complex(0, -1); // i^3 = -i (transition back)
    }
    
    // Calculate denominator: i^(|z|) + 1
    const denominator = iPowResult.add(new Complex(1, 0));
    
    // Store the current state in the path
    path.push({
      z: {real: z.real, imag: z.imag},
      magnitude: z.abs(),
      state: getTheologicalState(iPower)
    });
    
    try {
      // Apply the Trinitarian Mandelbrot equation:
      // z_{n+1} = (z_n³ + z_n² + z_n + c) / (i^(|z_n|) + 1)
      z = numerator.divide(denominator);
      
      // Check for escape
      if (z.abs() > escapeRadius) {
        return {
          escaped: true,
          iterations: iter + 1,
          finalZ: {real: z.real, imag: z.imag},
          path: path
        };
      }
    } catch (e) {
      // Division by zero represents a singularity (theological significance!)
      return {
        escaped: "singularity",
        iterations: iter + 1,
        path: path,
        error: e.message
      };
    }
  }
  
  // Point is in the set (didn't escape)
  return {
    escaped: false,
    iterations: maxIterations,
    finalZ: {real: z.real, imag: z.imag},
    path: path
  };
}

/**
 * Maps the power of i to its theological meaning
 * @param {number} iPower - The power of i (mod 4)
 * @returns {string} The theological state
 */
function getTheologicalState(iPower) {
  switch(iPower) {
    case 0: return "Identity/Being (Father)";
    case 1: return "Transition (Son)";
    case 2: return "Death/Negation";
    case 3: return "Return/Resurrection (Spirit)";
    default: return "Unknown";
  }
}

/**
 * Analyze key theological points using the Trinitarian Mandelbrot equation
 */
function analyzeTheologicalPoints() {
  const testPoints = [
    { name: "God (Unity/Truth)", value: new Complex(0, 0) },
    { name: "Father (Identity)", value: new Complex(1, 0) },
    { name: "Son (Non-Contradiction)", value: new Complex(2, 0) },
    { name: "Spirit (Excluded Middle)", value: new Complex(3, 0) },
    { name: "Resurrection Operator", value: new Complex(0, 1) },
    { name: "Death", value: new Complex(-1, 0) },
    { name: "Privation", value: new Complex(0, -1) }
  ];
  
  console.log("Trinitarian Mandelbrot Analysis:");
  
  testPoints.forEach(point => {
    console.log(`\nAnalyzing point: ${point.name} (${point.value})`);
    const result = trinitarianMandelbrot(point.value, 20, 4);
    
    if (result.escaped === "singularity") {
      console.log(`- Resulted in a singularity after ${result.iterations} iterations`);
    } else if (result.escaped) {
      console.log(`- Escaped after ${result.iterations} iterations`);
      console.log(`- Final value: ${result.finalZ.real}${result.finalZ.imag >= 0 ? '+' : ''}${result.finalZ.imag}i`);
    } else {
      console.log(`- Did not escape after ${result.iterations} iterations`);
      console.log(`- Final value: ${result.finalZ.real}${result.finalZ.imag >= 0 ? '+' : ''}${result.finalZ.imag}i`);
    }
    
    // Show first few iterations
    console.log("- Iteration pattern:");
    const showIterations = Math.min(5, result.path.length);
    for (let i = 0; i < showIterations; i++) {
      const p = result.path[i];
      console.log(`  Iter ${i}: z = ${p.z.real.toFixed(4)}${p.z.imag >= 0 ? '+' : ''}${p.z.imag.toFixed(4)}i, |z| = ${p.magnitude.toFixed(4)}, State: ${p.state}`);
    }
    
    if (result.path.length > 5) {
      console.log("  ...");
      const last = result.path[result.path.length - 1];
      console.log(`  Iter ${result.path.length - 1}: z = ${last.z.real.toFixed(4)}${last.z.imag >= 0 ? '+' : ''}${last.z.imag.toFixed(4)}i, |z| = ${last.magnitude.toFixed(4)}, State: ${last.state}`);
    }
  });
}

// Execute the analysis
analyzeTheologicalPoints();

/**
 * Generate data for plotting the Trinitarian Mandelbrot set
 * @param {number} width - Width of the grid
 * @param {number} height - Height of the grid
 * @param {number} xMin - Minimum x value
 * @param {number} xMax - Maximum x value
 * @param {number} yMin - Minimum y value
 * @param {number} yMax - Maximum y value
 * @param {number} maxIterations - Maximum iterations to perform
 * @returns {Array} Grid of iteration counts
 */
function generateMandelbrotData(width, height, xMin, xMax, yMin, yMax, maxIterations = 100) {
  const data = new Array(height);
  
  for (let y = 0; y < height; y++) {
    data[y] = new Array(width);
    
    for (let x = 0; x < width; x++) {
      // Map pixel coordinates to complex plane
      const real = xMin + (xMax - xMin) * x / (width - 1);
      const imag = yMin + (yMax - yMin) * y / (height - 1);
      
      // Calculate Trinitarian Mandelbrot value
      const c = new Complex(real, imag);
      const result = trinitarianMandelbrot(c, maxIterations, 4);
      
      // Store iteration count or special value for the set
      data[y][x] = result.escaped ? result.iterations : -1;
    }
  }
  
  return data;
}

/**
 * The final Trinitarian Mandelbrot equation:
 *
 * z_{n+1} = (z_n³ + z_n² + z_n + c) / (i^(|z_n| mod 4) + 1)
 *
 * This equation incorporates:
 * 1. The trinitarian structure (Father, Son, Spirit) as powers (z, z², z³)
 * 2. The powers of i representing ontological states:
 *    - i^0 = 1: Identity/Being (Father)
 *    - i^1 = i: Transition (Son)
 *    - i^2 = -1: Death/Negation
 *    - i^3 = -i: Return/Resurrection (Spirit)
 * 3. The unity-in-trinity relationship through division
 * 4. The Mandelbrot seed value c represents the starting point for theological exploration
 */