EQUIPMENT_DATA = [
  {
    "equipment": "Gas Insulated Switchgear (GIS)",
    "tools": ["Gas leak detector", "IR camera", "Contact resistance tester", "Multimeter"],
    "preventive_maintenance": [
      "Check SF6 gas pressure",
      "Inspect seals and gaskets",
      "Test insulation resistance",
      "Monitor partial discharge levels"
    ],
    "failure_symptoms": ["Gas leakage", "Flashover inside compartments", "Abnormal heating"],
    "safety_precautions": ["Ventilate room", "Wear gas mask", "De-energize before maintenance"]
  },
  {
    "equipment": "Shunt Reactor",
    "tools": ["Insulation resistance tester", "Oil testing kit", "Thermographic camera"],
    "preventive_maintenance": [
      "Check oil level and condition",
      "Perform insulation resistance tests",
      "Inspect bushings for cracks"
    ],
    "failure_symptoms": ["Overheating", "Unusual vibration", "Oil leakage"],
    "safety_precautions": ["De-energize and ground", "Wear PPE", "Handle bushings carefully"]
  },
  {
    "equipment": "Neutral Grounding Resistor (NGR)",
    "tools": ["Multimeter", "Resistance tester", "Infrared thermometer"],
    "preventive_maintenance": [
      "Measure resistance value",
      "Check for overheating marks",
      "Inspect connections"
    ],
    "failure_symptoms": ["High resistance", "Overheating", "Open circuit"],
    "safety_precautions": ["Isolate circuit", "Use insulated tools", "Ground properly"]
  },
  {
    "equipment": "Bus Coupler",
    "tools": ["Torque wrench", "Contact cleaner", "Thermal scanner"],
    "preventive_maintenance": [
      "Verify proper operation of coupler breaker",
      "Check interlocking system",
      "Clean contacts"
    ],
    "failure_symptoms": ["Failure to transfer load", "Excessive heating", "Contact wear"],
    "safety_precautions": ["Follow lockout/tagout procedure", "Use insulated gloves"]
  },
  {
    "equipment": "Waveguide for Communication",
    "tools": ["Network analyzer", "Signal tester", "Cleaning kit"],
    "preventive_maintenance": [
      "Check signal attenuation",
      "Inspect connectors",
      "Clean dust and moisture"
    ],
    "failure_symptoms": ["Signal loss", "Moisture ingress", "Loose connectors"],
    "safety_precautions": ["Avoid bending cables", "Ensure grounding"]
  },
  {
    "equipment": "Overhead Line Conductor",
    "tools": ["Binoculars", "Tension gauge", "Thermal scanner"],
    "preventive_maintenance": [
      "Inspect for sagging",
      "Check for bird nesting or flashover marks",
      "Scan hot spots"
    ],
    "failure_symptoms": ["Broken strands", "Excessive sag", "Flashover damage"],
    "safety_precautions": ["Isolate line", "Maintain safe clearance", "Avoid work in rain"]
  },
  {
    "equipment": "Pole Structure",
    "tools": ["Measuring tape", "Plumb line", "Hammer test tool"],
    "preventive_maintenance": [
      "Check foundation integrity",
      "Inspect for rust and cracks",
      "Ensure vertical alignment"
    ],
    "failure_symptoms": ["Leaning structure", "Rust patches", "Concrete cracks"],
    "safety_precautions": ["Wear helmet and harness", "Avoid climbing energized structures"]
  },
  {
    "equipment": "HVDC Converter Valve",
    "tools": ["Oscilloscope", "Multimeter", "Cooling system tester"],
    "preventive_maintenance": [
      "Check thyristor firing angle",
      "Inspect cooling system",
      "Test valve insulation"
    ],
    "failure_symptoms": ["Valve short circuit", "Overheating", "Harmonic distortion"],
    "safety_precautions": ["Isolate DC side", "Discharge capacitors", "Use insulated PPE"]
  },
  {
    "equipment": "Optical Ground Wire (OPGW)",
    "tools": ["OTDR tester", "Fiber cleaver", "Splicing kit"],
    "preventive_maintenance": [
      "Test fiber attenuation",
      "Inspect cable sheath",
      "Perform splicing quality check"
    ],
    "failure_symptoms": ["Signal loss", "Broken fiber strands", "Water ingress"],
    "safety_precautions": ["Avoid excessive bending", "Handle with fiber gloves"]
  },
  {
    "equipment": "High Voltage Surge Capacitor Coupler",
    "tools": ["Capacitance meter", "Megger", "IR tester"],
    "preventive_maintenance": [
      "Check capacitance values",
      "Inspect bushings",
      "Test insulation resistance"
    ],
    "failure_symptoms": ["Capacitance drift", "Insulation failure", "Overheating"],
    "safety_precautions": ["Ground before testing", "Wear insulating gloves"]
  },
  {
    "equipment": "Tap Changer (OLTC)",
    "tools": ["Oil sampling kit", "Torque wrench", "Multimeter"],
    "preventive_maintenance": [
      "Check oil quality in diverter switch",
      "Inspect mechanical alignment",
      "Test electrical contacts"
    ],
    "failure_symptoms": ["Arcing noise", "Uneven voltage regulation", "Oil contamination"],
    "safety_precautions": ["De-energize transformer", "Ground OLTC mechanism"]
  },
  {
    "equipment": "Fire Suppression System",
    "tools": ["Pressure gauge", "Smoke detector tester", "Extinguisher refill kit"],
    "preventive_maintenance": [
      "Check cylinder pressure",
      "Test smoke detectors",
      "Verify sprinkler nozzles"
    ],
    "failure_symptoms": ["Low pressure", "Clogged nozzle", "Failure to activate"],
    "safety_precautions": ["Keep exit clear", "Test in controlled mode"]
  },
  {
    "equipment": "HV Testing Transformer",
    "tools": ["High-voltage probe", "Voltmeter", "Protective resistor set"],
    "preventive_maintenance": [
      "Check insulation resistance",
      "Calibrate output voltage",
      "Inspect bushings"
    ],
    "failure_symptoms": ["Inaccurate test readings", "Flashover", "Insulation failure"],
    "safety_precautions": ["Use safety interlocks", "Ground secondary winding"]
  },
  {
    "equipment": "Station Lighting System",
    "tools": ["Lux meter", "Screwdriver set", "Multimeter"],
    "preventive_maintenance": [
      "Check lamp health",
      "Inspect wiring",
      "Test emergency lights"
    ],
    "failure_symptoms": ["Dark spots", "Flickering lights", "Burnt wiring"],
    "safety_precautions": ["De-energize circuit", "Use insulated ladders"]
  },
  {
    "equipment": "Control Room HVAC System",
    "tools": ["Thermometer", "Refrigerant gauge", "Filter cleaning kit"],
    "preventive_maintenance": [
      "Clean air filters",
      "Check refrigerant pressure",
      "Inspect compressor health"
    ],
    "failure_symptoms": ["Poor cooling", "Excessive noise", "High humidity"],
    "safety_precautions": ["Switch off before maintenance", "Avoid refrigerant exposure"]
  },
  {
    "equipment": "Station Alarm System",
    "tools": ["Siren tester", "Battery tester", "Multimeter"],
    "preventive_maintenance": [
      "Test alarm function monthly",
      "Check battery backup",
      "Inspect wiring"
    ],
    "failure_symptoms": ["Alarm not sounding", "False alarms", "Battery low"],
    "safety_precautions": ["Test in non-critical hours", "Document alarm triggers"]
  },
  {
    "equipment": "SCADA System",
    "tools": ["Laptop with SCADA software", "Network analyzer", "Multimeter"],
    "preventive_maintenance": [
      "Update SCADA software",
      "Test communication links",
      "Check data accuracy"
    ],
    "failure_symptoms": ["Data loss", "Communication lag", "False alarms"],
    "safety_precautions": ["Back up data before updates", "Use secure login"]
  },
  {
    "equipment": "AC Distribution Panel",
    "tools": ["Multimeter", "Torque wrench", "IR camera"],
    "preventive_maintenance": [
      "Inspect fuses and breakers",
      "Tighten loose connections",
      "Thermal scan for hot spots"
    ],
    "failure_symptoms": ["Fuse blowing", "Breaker tripping", "Overheating"],
    "safety_precautions": ["Lockout/tagout procedure", "Avoid overload testing"]
  },
  {
    "equipment": "DC Distribution Board",
    "tools": ["Multimeter", "Insulation tester", "Screwdriver set"],
    "preventive_maintenance": [
      "Check voltage output",
      "Inspect wiring",
      "Test fuse condition"
    ],
    "failure_symptoms": ["No DC supply", "Blown fuses", "Loose connections"],
    "safety_precautions": ["De-energize before testing", "Label polarities"]
  },
  {
    "equipment": "Station Earthing Mat",
    "tools": ["Earth tester", "Clamp meter", "Soil resistivity tester"],
    "preventive_maintenance": [
      "Measure grid resistance",
      "Check for corrosion",
      "Verify continuity of earth mat"
    ],
    "failure_symptoms": ["High step potential", "Ground faults not clearing", "Corrosion"],
    "safety_precautions": ["Test in dry/wet seasons", "Avoid maintenance during thunderstorms"]
  },
  {
    "equipment": "Power Transformer",
    "tools": ["Oil testing kit", "IR tester", "Thermographic camera", "Wrenches", "Torque meter"],
    "preventive_maintenance": [
      "Check oil level and top up",
      "Perform Dissolved Gas Analysis (DGA) every 6 months",
      "Inspect bushings for cracks or leaks",
      "Tighten loose connections with torque meter"
    ],
    "failure_symptoms": ["Overheating", "Oil discoloration", "Abnormal noise", "Tripping on protection"],
    "safety_precautions": ["Ensure transformer is de-energized", "Wear arc-flash PPE", "Ground equipment before maintenance"]
  },
  {
    "equipment": "Circuit Breaker (SF6 / Vacuum)",
    "tools": ["SF6 gas detector", "Contact resistance tester", "Pressure gauge", "Multimeter"],
    "preventive_maintenance": [
      "Measure contact resistance annually",
      "Check SF6 pressure levels",
      "Clean arc contacts and lubricate moving parts",
      "Verify proper tripping & closing operation"
    ],
    "failure_symptoms": ["Incomplete operation", "Abnormal gas pressure", "Excessive heating"],
    "safety_precautions": ["Ventilate area for SF6 handling", "Discharge stored energy", "Isolate breaker completely"]
  },
  {
    "equipment": "Current Transformer (CT)",
    "tools": ["Insulation resistance tester", "Primary injection kit", "Megger", "Multimeter"],
    "preventive_maintenance": [
      "Perform insulation resistance test annually",
      "Check secondary connections for loose contacts",
      "Verify ratio accuracy with test kit"
    ],
    "failure_symptoms": ["Inaccurate readings", "Overheating", "Partial discharge sound"],
    "safety_precautions": ["Short secondary terminals before disconnection", "Use insulated tools"]
  },
  {
    "equipment": "Potential Transformer (PT)",
    "tools": ["Insulation tester", "Multimeter", "Secondary injection kit"],
    "preventive_maintenance": [
      "Check insulation resistance yearly",
      "Inspect bushings for cracks",
      "Verify voltage ratio"
    ],
    "failure_symptoms": ["Incorrect voltage readings", "Bushing damage", "Abnormal heating"],
    "safety_precautions": ["De-energize before testing", "Ground secondary winding before disconnection"]
  },
  {
    "equipment": "Lightning Arrester",
    "tools": ["Leakage current meter", "IR tester", "Clamp meter"],
    "preventive_maintenance": [
      "Measure leakage current every 6 months",
      "Inspect for physical damage or cracks",
      "Ensure earthing resistance < 1 ohm"
    ],
    "failure_symptoms": ["Frequent flashovers", "Cracks in housing", "High leakage current"],
    "safety_precautions": ["Test during dry conditions", "Handle with insulated gloves"]
  },
  {
    "equipment": "Busbars & Connectors",
    "tools": ["Infrared thermal camera", "Wrenches", "Torque meter", "Contact cleaner"],
    "preventive_maintenance": [
      "Inspect for corrosion or discoloration",
      "Perform thermographic scanning to detect hot spots",
      "Tighten connections periodically"
    ],
    "failure_symptoms": ["Excessive heating", "Sparking", "Burnt smell"],
    "safety_precautions": ["Isolate section before maintenance", "Ensure proper grounding"]
  },
  {
    "equipment": "Battery Bank & Charger",
    "tools": ["Battery hydrometer", "Voltmeter", "Load tester", "Distilled water"],
    "preventive_maintenance": [
      "Check electrolyte level and top up with distilled water",
      "Measure specific gravity monthly",
      "Verify charger voltage & current settings"
    ],
    "failure_symptoms": ["Low backup time", "Bulging cells", "High self-discharge"],
    "safety_precautions": ["Neutralize spills with baking soda", "Wear acid-resistant gloves & face shield"]
  },
  {
    "equipment": "Protective Relays",
    "tools": ["Secondary injection test kit", "Multimeter", "Relay testing software"],
    "preventive_maintenance": [
      "Test relay functionality annually",
      "Verify tripping times and settings",
      "Clean contacts and check wiring integrity"
    ],
    "failure_symptoms": ["False tripping", "Relay not operating", "Sluggish operation"],
    "safety_precautions": ["Use isolation before secondary injection", "Document all settings before modification"]
  },
  {
    "equipment": "Earthing System",
    "tools": ["Earth resistance tester", "Clamp-on ground tester", "Spade"],
    "preventive_maintenance": [
      "Measure earth resistance annually",
      "Check for corrosion of electrodes",
      "Ensure tight connections to grounding conductors"
    ],
    "failure_symptoms": ["High resistance", "Sparking", "Ineffective fault clearance"],
    "safety_precautions": ["Test in dry and wet seasons", "Avoid working during thunderstorms"]
  },
  {
    "equipment": "Isolator/Disconnector",
    "tools": ["Torque wrench", "Contact cleaner", "Lubrication kit"],
    "preventive_maintenance": [
      "Check alignment of blades",
      "Clean and lubricate moving parts",
      "Inspect contacts for wear"
    ],
    "failure_symptoms": ["Incomplete opening/closing", "Corroded contacts", "High contact resistance"],
    "safety_precautions": ["Operate only when circuit is de-energized", "Use insulated rods"]
  },
  {
    "equipment": "Control Cables",
    "tools": ["Cable tester", "Multimeter", "Insulation tester"],
    "preventive_maintenance": [
      "Inspect insulation condition",
      "Test continuity and insulation resistance",
      "Check for rodent damage"
    ],
    "failure_symptoms": ["Signal loss", "Short circuit", "Open circuit faults"],
    "safety_precautions": ["De-energize before testing", "Label cables during maintenance"]
  },
  {
    "equipment": "Auxiliary Transformer",
    "tools": ["Insulation tester", "Voltmeter", "Thermal scanner"],
    "preventive_maintenance": [
      "Check winding resistance",
      "Inspect cooling system",
      "Measure insulation resistance annually"
    ],
    "failure_symptoms": ["Overheating", "Voltage fluctuations", "Abnormal vibration"],
    "safety_precautions": ["Isolate before maintenance", "Ground secondary"]
  },
  {
    "equipment": "Capacitor Bank",
    "tools": ["Capacitance meter", "Insulation tester", "Infrared camera"],
    "preventive_maintenance": [
      "Measure capacitance value annually",
      "Check for bulging or leakage",
      "Ensure balancing reactor condition"
    ],
    "failure_symptoms": ["Unbalanced load", "Excessive heating", "Explosion risk"],
    "safety_precautions": ["Discharge capacitors before touching", "Use insulated tools"]
  },
  {
    "equipment": "Control Panel",
    "tools": ["Multimeter", "Contact cleaner", "Screwdriver set"],
    "preventive_maintenance": [
      "Clean dust and dirt periodically",
      "Check wiring tightness",
      "Test indicators and alarms"
    ],
    "failure_symptoms": ["Faulty indicators", "Loose connections", "Panel overheating"],
    "safety_precautions": ["De-energize before opening panel", "Avoid water ingress"]
  },
  {
    "equipment": "Insulators",
    "tools": ["Binoculars", "IR camera", "Cleaning tools"],
    "preventive_maintenance": [
      "Inspect for cracks or flashover marks",
      "Clean insulators in polluted areas",
      "Check alignment"
    ],
    "failure_symptoms": ["Cracks", "Surface discharge", "Loss of mechanical strength"],
    "safety_precautions": ["Do not climb energized structures", "Use live-line tools if required"]
  },
  {
    "equipment": "Surge Capacitor",
    "tools": ["Capacitance meter", "Insulation tester"],
    "preventive_maintenance": [
      "Check capacitance value",
      "Inspect bushings",
      "Test insulation resistance"
    ],
    "failure_symptoms": ["Capacitance drift", "Insulation failure", "Heating"],
    "safety_precautions": ["Isolate before testing", "Ground terminals before handling"]
  },
  {
    "equipment": "Wave Trap (Line Trap)",
    "tools": ["Multimeter", "Oscilloscope", "Tuning kit"],
    "preventive_maintenance": [
      "Check tuning frequency",
      "Inspect coils and connections",
      "Clean dust and rust"
    ],
    "failure_symptoms": ["Communication failure", "Heating", "Loose connections"],
    "safety_precautions": ["De-energize line before work", "Use insulated gloves"]
  },
  {
    "equipment": "Diesel Generator (DG Set)",
    "tools": ["Multimeter", "Fuel tester", "Oil level gauge", "Tool kit"],
    "preventive_maintenance": [
      "Check fuel level and quality",
      "Inspect oil and coolant levels",
      "Run test load every month"
    ],
    "failure_symptoms": ["Failure to start", "Low voltage output", "Excessive smoke"],
    "safety_precautions": ["Keep fire extinguisher nearby", "Ventilate exhaust gases"]
  },
  {
    "equipment": "HV Cables",
    "tools": ["High-voltage tester", "Insulation tester", "Sheath tester"],
    "preventive_maintenance": [
      "Check insulation resistance",
      "Inspect joints and terminations",
      "Perform partial discharge test"
    ],
    "failure_symptoms": ["Breakdown", "Surface tracking", "Excessive heating"],
    "safety_precautions": ["Discharge stored charge", "Use PPE"]
  },
  {
    "equipment": "Cooling System (Fans, Pumps, Radiators)",
    "tools": ["Thermal scanner", "Flow meter", "Cleaning brushes"],
    "preventive_maintenance": [
      "Check fan and pump operation",
      "Clean radiator fins",
      "Inspect coolant level"
    ],
    "failure_symptoms": ["Overheating equipment", "Noise in pumps", "Reduced airflow"],
    "safety_precautions": ["Isolate motor supply", "Avoid working near hot surfaces"]
  },

  {
    "equipment": "Grounding Transformer",
    "tools": ["Insulation tester", "Multimeter", "Thermal camera"],
    "preventive_maintenance": [
      "Check insulation resistance annually",
      "Inspect bushings and connections",
      "Measure winding resistance"
    ],
    "failure_symptoms": ["High neutral voltage", "Overheating", "Abnormal humming"],
    "safety_precautions": ["De-energize and ground before inspection", "Use insulated gloves"]
  },
  {
    "equipment": "Station Auxiliary Diesel Pump",
    "tools": ["Pressure gauge", "Wrenches", "Oil tester"],
    "preventive_maintenance": [
      "Check oil level and quality",
      "Test pump operation under load",
      "Inspect hoses and valves"
    ],
    "failure_symptoms": ["Low pressure", "Leakage", "Failure to start"],
    "safety_precautions": ["Keep fire extinguisher nearby", "Avoid running dry"]
  },
  {
    "equipment": "Remote Terminal Unit (RTU)",
    "tools": ["Laptop with configuration software", "Multimeter", "Ethernet tester"],
    "preventive_maintenance": [
      "Check firmware updates",
      "Test communication links",
      "Verify I/O signals"
    ],
    "failure_symptoms": ["Data communication failure", "No response", "Incorrect readings"],
    "safety_precautions": ["Backup configurations before update", "Use ESD protection"]
  },
  {
    "equipment": "Automatic Voltage Regulator (AVR)",
    "tools": ["Oscilloscope", "Multimeter", "Test bench"],
    "preventive_maintenance": [
      "Verify voltage regulation accuracy",
      "Inspect control circuitry",
      "Clean dust from PCB"
    ],
    "failure_symptoms": ["Voltage fluctuations", "Erratic operation", "Failure to regulate"],
    "safety_precautions": ["Isolate supply before inspection", "Use antistatic precautions"]
  },
  {
    "equipment": "Station Communication Router",
    "tools": ["Network analyzer", "Ethernet tester", "Laptop"],
    "preventive_maintenance": [
      "Update firmware",
      "Test network throughput",
      "Check port integrity"
    ],
    "failure_symptoms": ["Network downtime", "Packet loss", "Overheating"],
    "safety_precautions": ["Use secure credentials", "Backup configs before change"]
  },
  {
    "equipment": "Voltage Regulating Relay",
    "tools": ["Secondary injection kit", "Multimeter", "Calibration tools"],
    "preventive_maintenance": [
      "Test relay settings annually",
      "Inspect wiring",
      "Calibrate voltage sensing circuit"
    ],
    "failure_symptoms": ["Incorrect voltage regulation", "Relay not operating", "False trips"],
    "safety_precautions": ["Isolate circuit before testing", "Record all settings"]
  },
  {
    "equipment": "Station Fire Alarm Panel",
    "tools": ["Smoke detector tester", "Battery tester", "Multimeter"],
    "preventive_maintenance": [
      "Test detectors quarterly",
      "Check backup batteries",
      "Inspect panel indicators"
    ],
    "failure_symptoms": ["No alarm activation", "False alarms", "Dead backup battery"],
    "safety_precautions": ["Test alarms during non-peak hours", "Notify control room before testing"]
  },
  {
    "equipment": "Transformer Cooling Oil Pump",
    "tools": ["Flow meter", "Multimeter", "Lubrication kit"],
    "preventive_maintenance": [
      "Check oil flow rate",
      "Inspect pump bearings",
      "Lubricate moving parts"
    ],
    "failure_symptoms": ["Low oil circulation", "Overheating transformer", "Noise from pump"],
    "safety_precautions": ["Isolate motor supply", "Avoid contact with hot oil"]
  },
  {
    "equipment": "GIS Local Control Cabinet",
    "tools": ["Multimeter", "Contact cleaner", "Screwdriver set"],
    "preventive_maintenance": [
      "Check wiring tightness",
      "Test control switches",
      "Clean dust inside cabinet"
    ],
    "failure_symptoms": ["Control failure", "Faulty indications", "Loose wiring"],
    "safety_precautions": ["De-energize before inspection", "Close doors securely after work"]
  },
  {
    "equipment": "Transformer Buchholz Relay",
    "tools": ["Gas sampling kit", "Multimeter", "Relay tester"],
    "preventive_maintenance": [
      "Check relay operation",
      "Test alarm and trip contacts",
      "Inspect for gas accumulation"
    ],
    "failure_symptoms": ["False alarms", "Relay not tripping", "Gas bubble formation"],
    "safety_precautions": ["Handle gas samples carefully", "Test during outage if possible"]
  }
]

