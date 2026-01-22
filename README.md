# Wine_Quality_Evaluation

This is project developed for the FMI course "Ineteligent Agents with Generative Artificial Inteligence" 

Project name	DionysusAi – Intelligent Wine Quality Assessment system 

Short project description (Business needs and system features)
In the wine industry and among enthusiasts, consistent high quality assessment requires expert knowledge but in the end it often end up subjective. The purpose of this project is to develop a multi-agent system that combines chemical sensors, computer vision, and machine learning to provide objective and easy to comprehend wine quality assessments with integrated market intelligence and personalized education. The system simulates a professional sommelier's evaluation through multi-modal data analysis and provides educational inquiries for users, as well as an active market analysis and investment guidance.
The system, named Dionysus, comprises of:
1.	Sensor Fusion Agent (SFA): Collects and preprocesses data from multiple sensors (pH, TDS, color, temperature) using MQTT for real-time communication.
2.	Quality Prediction Agent (QPA): Ensemble machine learning model (Random Forest & XGBoost) that predicts wine quality scores (0-10) using the UCI Wine Quality Dataset.
3.	Visual Analysis Agent (VAA): Computer vision system using OpenCV and YOLO for analyzing wine color, clarity, and appearance through camera input.
4.	Explanation Agent (EA): Generates human-readable reports using SHAP values and provides food pairing recommendations whilst impersonating the character of a sommelier.
5.	User Interface Agent (UIA): Manages web dashboard (React), voice interface (Speech-to-Text and Text-to-Speech).
6.	Wine Mentor Agent (WMA): Fine-tuned LLM that acts as a personal sommelier, providing interactive wine education, tasting guidance, and adaptive learning experiences.
Data persistence uses SQLite/PostgreSQL, with real-time sensor data streaming via WebSocket. Inter-agent communication employs gRPC for high-performance, bidirectional exchanges between specialized agents.



2.	ML/Agent System Description using PEAS [https://aima.cs.berkeley.edu/4th-ed/pdfs/newchap02.pdf]

Agent name	Performance Measure	Environment	Actuators/Outputs	Sensors/Inputs
1. SFA	Accurate sensor data fusion; Minimal data loss; Fast real-time processing	Laboratory environment with multiple chemical sensors (pH, TDS, color); Variable environmental conditions (temperature, humidity)	Send processed sensor data streams to QPA and EA using gRPC; Send calibration commands to sensors using MQTT.	Receive real-time data from pH sensor, TDS sensor, RGB color sensor, temperature sensor via MQTT; Receive configuration updates from UIA via gRPC.
2. QPA	High prediction accuracy (>85%) vs expert ratings; Fast inference time (<2s); Robust to sensor noise	 Server environment;
running in a docker container; connected with other agents using gRPC.	Send quality predictions with confidence scores to EA and UIA using gRPC; Send model performance metrics to EA.	Receive fused sensor data from SFA via gRPC; Receive model retraining requests from EA via gRPC.
3. VAA	Accurate visual feature extraction; Consistent color analysis; Fast image processing (<1s)	Controlled lighting environment with camera setup; running in a docker container; connected with other agents using gRPC.	 Send visual analysis results (color analysis, clarity scores) to QPA and EA using gRPC.	Receive image streams from camera module via WebSocket; Receive analysis requests from UIA via gRPC.
4. EA	Clear, actionable explanations; Useful pairing recommendations; High user satisfaction scores	 Server environment; running in a docker container; connected with other agents using gRPC.	Send comprehensive analysis reports to UIA using gRPC; Send SHAP explanation plots and charts; Provide food pairing recommendations; 	Receive quality predictions from QPA; Receive visual data from VAA; Receive user queries from UIA via gRPC.
5. MIA	Accurate market trend predictions (>80% accuracy); High return on investment recommendations; Real-time price alert precision	Connected to external wine market APIs and financial data sources; running in a docker container; connected with other agents using gRPC.	Send market analysis reports and investment recommendations to UIA using gRPC; Provide price alerts and arbitrage opportunities.	Receive external market data from wine auction APIs; Receive wine quality data from QPA; Receive user investment preferences from UIA via gRPC.
6. WMA	Educational effectiveness; User knowledge improvement rates; Engagement and satisfaction scores; Adaptive learning precision	Server environment with GPU acceleration for LLM inference; running in a docker container; connected with other agents using gRPC.	Send interactive educational content, tasting guidance, and knowledge assessments to UIA using gRPC; Provide personalized learning recommendations.	Receive user knowledge level and learning goals from UIA; Receive wine analysis data from EA and QPA for contextual education; Receive educational content requests via gRPC.
7. UIA	 Optimal user experience; Responsive web interface; Accurate speech recognition; Real-time data visualization	Multimedia computing device with speakers, microphone, camera, screen; Connected to sensors and other agents.	 Display web dashboard with real-time data visualization; Play Text-to-Speech audio output; Send user commands to other agents via gRPC.	Receive user voice commands via Speech-to-Text; Receive web interface interactions; Receive sensor status from SFA via gRPC.

3.	Main Use Cases / Scenarios
Use case name	Brief Descriptions	Actors Involved
1.	Single Sample Quality Assessment	The User places a wine sample in the testing chamber and initiates analysis via voice or web UI (UIA). System performs automated sensor readings (SFA), visual analysis (VAA), quality prediction (QPA), and generates comprehensive report (EA).	User, SFA, QPA, VAA, EA, UIA
2. Comparative Wine Analysis	The User tests multiple wine samples sequentially. System provides comparative report with rankings and highlights key differentiating factors between samples for educational or purchasing decisions.	User, SFA, QPA, VAA, EA, UIA
3. Food Pairing Recommendation 	The User inputs a planned meal or selects from database. System suggests optimal wine pairings based on wine characteristics and provides explanations for the recommendations.	User, EA, UIA
…4 4. Batch Processing & Quality Control
	The User processes multiple production samples. System generates consolidated quality report, identifies trends and outliers, and provides production batch insights.	 User, SFA, QPA, VAA, EA, UIA
       5. System Calibration & Maintenance 
	Administrator performs sensor calibration and model retraining using reference samples. System provides calibration reports and performance metrics.	Administrator, SFA,UIA
       6. Investment Opportunity Analysis 
	User requests market analysis for specific wine or region. MIA analyzes current market trends, historical performance, and quality predictions to provide investment recommendations and price alerts.	User, MIA, QPA, UIA
		
8. Interactive Wine Education	User engages in conversational learning with WMA about wine regions, tasting techniques, production methods, and sensory analysis with adaptive difficulty.	User, WMA, UIA
9. Virtual Tasting Session 	WMA guides user through structured tasting exercises, comparing multiple wines with real-time sensor analysis and educational commentary.	User, WMA, SFA, QPA, VAA, EA, UIA

4.	API Resources (REST/SSE/WebSocket Backend)
View name	Brief Descriptions	URI
1. Samples 
	POST new wine sample analysis request; GET analysis history for all samples. Available for authenticated users.	/api/samples
2. Analysis
	POST sensor data for real-time quality assessment; GET detailed analysis reports with SHAP explanations.	/api/analysis
3. Pairings 
	GET food pairing recommendations based on wine characteristics; POST custom pairing preferences.	/api/pairings
4. Sensors
	GET real-time sensor data streams; POST sensor calibration commands. 	/api/sensors
5. Models
	GET model performance metrics; POST model retraining requests with new labeled data. 	/api/models

6. Sensor Stream	WebSocket stream of real-time sensor data and analysis progress updates.	/ws/sensor-data
7. Visual Stream
	WebSocket stream of camera feed with computer vision overlays and analysis results.	/ws/visual-feed
8. Authentication	POST user credentials for login and receive security token; POST logout request to invalidate token.	/api/auth/login, /api/auth/logout
9. User Management 	GET, PUT, DELETE user profiles and preferences; POST new user registration with taste preference initialization.	/api/users/{userId}, /api/users/register
10. Market Analysis 	GET current wine market trends and price predictions; POST specific wine investment analysis requests; GET vintage performance reports.	/api/market/trends, /api/market/investment
11. Report 	POST request for comprehensive analysis report generation; GET exported reports in PDF/JSON format for professional use.	/api/reports/export
12. Education System	WebSocket for real-time interactive education with WMA. Supports voice interaction and adaptive content delivery	/ws/education-sessions
13. Wine Eductaion Sessions 	POST conversational education sessions with WMA. GET session history and transcripts. PUT user feedback on educational content	/api/education/sessions
