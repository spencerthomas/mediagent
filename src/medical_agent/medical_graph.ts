/**
 * Medical Diagnostic Agent Graph
 * Main workflow graph that orchestrates the medical superintelligence system
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";
// import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { 
  MedicalDiagnosticAnnotation, 
  MedicalDiagnosticStateType, 
  DiagnosticPhase 
} from "./medical_state.js";
import { ChainOfDebate } from "./chain_of_debate.js";
import { CaseInformationGatekeeper } from "./gatekeeper.js";
import { PatientQuestionAgent } from "./patient_question_agent.js";
// import { createInformationRequestTool } from "./medical_tools.js";
import { loadChatModel } from "../react_agent/utils.js";
import { ConfigurationSchema } from "../react_agent/configuration.js";
import { ensureMedicalConfiguration } from "./configuration.js";

// Initialize components
const gatekeeper = new CaseInformationGatekeeper(loadChatModel);
const patientQuestionAgent = new PatientQuestionAgent(loadChatModel);
const chainOfDebate = new ChainOfDebate(loadChatModel);

/**
 * Node: Initialize Case
 * Sets up the initial case information and state
 */
async function initializeCase(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🚨 initializeCase called");
  console.log("Current phase:", state.currentPhase);
  console.log("Message count:", state.messages.length);
  console.log("Awaiting user input:", state.awaitingUserInput);
  console.log("Pending questions:", state.pendingQuestions?.length || 0);
  
  // CRITICAL: If case is already initialized, DO NOT re-initialize
  // This prevents the infinite loop when resuming from interrupts
  if (state.availableCaseInfo?.patientId) {
    console.log("⚠️  Case already initialized with ID:", state.availableCaseInfo.patientId);
    console.log("🔄 This appears to be a resume from interrupt - skipping initialization");
    
    // Don't add any new messages or change state - just pass through
    return {};
  }
  
  // Initialize with basic case presentation
  const lastMessage = state.messages[state.messages.length - 1];
  const caseText = typeof lastMessage?.content === 'string' ? lastMessage.content : "";
  
  console.log("📝 Case text for extraction:", caseText.substring(0, 200));
  
  if (!caseText || caseText.trim().length === 0) {
    console.log("❌ No case text available, using fallback initialization");
    // Fallback initialization without LLM call
    const fallbackCaseInfo = {
      patientId: `CASE_${Date.now()}`,
      demographics: { age: 45, gender: 'unknown' },
      chiefComplaint: "Patient case presentation",
      historyOfPresentIllness: "Initial case presentation",
      pastMedicalHistory: [],
      medications: [],
      allergies: [],
      familyHistory: [],
      socialHistory: ""
    };
    
    return {
      availableCaseInfo: fallbackCaseInfo,
      currentPhase: 'initial_assessment',
      costBudget: 1000,
      messages: [
        ...state.messages,
        new AIMessage(`Case initialized: ${fallbackCaseInfo.patientId} - Please provide patient information`)
      ]
    };
  }
  
  try {
    // Extract basic case information from the input
    console.log("🤖 Calling extractCaseInformation...");
    const caseInfo = await extractCaseInformation(caseText, config);
    console.log("✅ Case info extracted:", caseInfo.patientId);
    
    return {
      availableCaseInfo: caseInfo,
      currentPhase: 'initial_assessment',
      costBudget: 1000, // Default budget
      messages: [
        ...state.messages,
        new AIMessage(`Case initialized: ${caseInfo.patientId} - ${caseInfo.chiefComplaint}`)
      ]
    };
  } catch (error) {
    console.error("❌ Error in extractCaseInformation:", error);
    // Fallback initialization on error
    const fallbackCaseInfo = {
      patientId: `CASE_${Date.now()}`,
      demographics: { age: 45, gender: 'unknown' },
      chiefComplaint: caseText.substring(0, 100),
      historyOfPresentIllness: caseText,
      pastMedicalHistory: [],
      medications: [],
      allergies: [],
      familyHistory: [],
      socialHistory: ""
    };
    
    return {
      availableCaseInfo: fallbackCaseInfo,
      currentPhase: 'initial_assessment',
      costBudget: 1000,
      messages: [
        ...state.messages,
        new AIMessage(`Case initialized with basic info: ${fallbackCaseInfo.patientId}`)
      ]
    };
  }
}

/**
 * Node: Medical Debate
 * Orchestrates the chain of debate between physician agents
 */
async function medicalDebate(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  return await chainOfDebate.orchestrateDebate(state, config);
}

/**
 * Node: Patient Interaction
 * Generates follow-up questions and handles patient responses
 */
async function patientInteraction(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  // If we have pending questions, wait for user input
  if (state.pendingQuestions.length > 0) {
    return {
      awaitingUserInput: true,
      messages: [
        ...state.messages,
        new AIMessage(`Please answer the following questions to help with your diagnosis:\n\n${
          state.pendingQuestions.map((q, i) => `${i + 1}. ${q.question}`).join('\n')
        }`)
      ]
    };
  }

  // Generate new questions if needed
  if (patientQuestionAgent.shouldGenerateQuestions(state)) {
    const questions = await patientQuestionAgent.generateQuestions(state, config);
    
    if (questions.length > 0) {
      const roundNumber = state.interactionRound + 1;
      return {
        pendingQuestions: questions,
        questionHistory: [...state.questionHistory, ...questions],
        awaitingUserInput: true,
        interactionRound: roundNumber,
        currentPhase: 'patient_interaction',
        messages: [
          ...state.messages,
          new AIMessage(`**Patient Interview - Round ${roundNumber}**\n\nI need some additional information to help with your diagnosis. Please answer the following questions:\n\n${
            questions.map((q, i) => `${i + 1}. ${q.question}`).join('\n')
          }`)
        ]
      };
    }
  }
  
  return {};
}

/**
 * Node: Process User Response
 * Processes user responses to patient questions
 */
async function processUserResponse(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  // This node is called when user provides responses
  // The actual response processing is handled by the gatekeeper
  const lastMessage = state.messages[state.messages.length - 1];
  
  if (lastMessage && typeof lastMessage.content === 'string') {
    // Process the user's response through the gatekeeper
    const updatedCaseInfo = await gatekeeper.processUserResponse(
      state,
      lastMessage.content,
      config
    );
    
    // Determine what information was gathered from this response
    try {
      const extractedInfo = JSON.parse(lastMessage.content);
      const updatedGathered = gatekeeper.identifyGatheredInformation(
        lastMessage.content,
        extractedInfo,
        state.requiredInformationGathered
      );
      
      return {
        availableCaseInfo: updatedCaseInfo,
        requiredInformationGathered: updatedGathered,
        pendingQuestions: [], // Clear pending questions
        awaitingUserInput: false,
        currentPhase: 'information_gathering', // Continue with analysis
        messages: [
          ...state.messages,
          new AIMessage(`Thank you for the additional information from Round ${state.interactionRound}. Let me analyze this with my medical team and determine if we need any clarification.`)
        ]
      };
    } catch {
      // Fallback for non-JSON responses
      const updatedGathered = gatekeeper.identifyGatheredInformation(
        lastMessage.content,
        {},
        state.requiredInformationGathered
      );
      
      return {
        availableCaseInfo: updatedCaseInfo,
        requiredInformationGathered: updatedGathered,
        pendingQuestions: [], // Clear pending questions
        awaitingUserInput: false,
        currentPhase: 'information_gathering', // Continue with analysis
        messages: [
          ...state.messages,
          new AIMessage(`Thank you for the additional information from Round ${state.interactionRound}. Let me analyze this with my medical team and determine if we need any clarification.`)
        ]
      };
    }
  }
  
  return {};
}

/**
 * Node: Diagnostic Tools
 * Executes medical diagnostic tools when needed
 */
async function executeDiagnosticTools(
  state: MedicalDiagnosticStateType,
  _config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  
  // Check if any agent has requested specific tools
  const lastAgentTurn = state.agentTurns[state.agentTurns.length - 1];
  
  if (lastAgentTurn?.testsRequested && lastAgentTurn.testsRequested.length > 0) {
    // Execute diagnostic test selector tool
    const toolMessage = new AIMessage("", {
      tool_calls: [{
        id: "diagnostic_test_selector",
        name: "diagnostic_test_selector",
        args: {
          differentialDiagnoses: state.differentialDiagnoses.map(d => d.condition),
          patientAge: state.availableCaseInfo.demographics.age,
          patientGender: state.availableCaseInfo.demographics.gender,
          chiefComplaint: state.availableCaseInfo.chiefComplaint,
          maxCost: state.costBudget - state.cumulativeCost
        }
      }]
    });
    
    // This would be processed by the tool node in a real implementation
    return {
      messages: [
        ...state.messages,
        toolMessage,
        new AIMessage("Diagnostic tools executed")
      ]
    };
  }
  
  return {};
}

/**
 * Node: Final Assessment
 * Generates final diagnostic assessment and recommendations
 */
async function finalAssessment(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  const configuration = ensureMedicalConfiguration(config);
  const model = await loadChatModel(configuration.model);
  
  const assessmentPrompt = `
Based on the medical diagnostic debate, provide a final assessment:

Case: ${state.availableCaseInfo.chiefComplaint}
Top Diagnoses: ${state.differentialDiagnoses.slice(0, 3).map(d => `${d.condition} (${d.probability}%)`).join(', ')}
Total Cost: $${state.cumulativeCost}
Confidence Level: ${(state.confidenceLevel * 100).toFixed(1)}%

Provide:
1. Primary diagnosis with confidence level
2. Alternative diagnoses to consider
3. Recommended next steps
4. Cost-effectiveness analysis
5. Quality of diagnostic process
`;

  const response = await model.invoke([new HumanMessage(assessmentPrompt)]);
  
  return {
    messages: [
      ...state.messages,
      new AIMessage(typeof response.content === 'string' ? response.content : 'Assessment completed')
    ],
    currentPhase: 'final_diagnosis' as DiagnosticPhase
  };
}

/**
 * Routing function to determine next step in workflow
 * NOTE: This should never route back to initialize_case - that only happens once at the start
 */
function routeWorkflow(state: MedicalDiagnosticStateType): string {
  // Check if awaiting user input
  if (state.awaitingUserInput) {
    return 'patient_interaction';
  }
  
  // PRIORITY: Check if patient interaction is needed (more aggressive)
  if (patientQuestionAgent.shouldGenerateQuestions(state)) {
    return 'patient_interaction';
  }
  
  // Check if final diagnosis is ready (only after sufficient patient interaction)
  if (state.readyForDiagnosis || state.currentPhase === 'final_diagnosis') {
    // Ensure we've had enough patient interaction before final diagnosis
    if (state.interactionRound < 3 && state.confidenceLevel < 0.8) {
      return 'patient_interaction';
    }
    return 'final_assessment';
  }
  
  // Check if over budget
  if (state.cumulativeCost >= state.costBudget) {
    return 'final_assessment';
  }
  
  // Check if maximum debate rounds reached (but still allow patient interaction)
  if (state.debateRound >= 5) {
    if (state.interactionRound < 3 || patientQuestionAgent.shouldGenerateQuestions(state)) {
      return 'patient_interaction';
    }
    return 'final_assessment';
  }
  
  // Force patient interaction after every 2 debate rounds if we haven't had enough
  if (state.debateRound > 0 && state.debateRound % 2 === 0 && state.interactionRound < 3) {
    return 'patient_interaction';
  }
  
  // Check if tools are needed
  const lastAgentTurn = state.agentTurns[state.agentTurns.length - 1];
  if (lastAgentTurn?.testsRequested && lastAgentTurn.testsRequested.length > 0) {
    return 'diagnostic_tools';
  }
  
  // Continue with medical debate
  return 'medical_debate';
}

/**
 * Routing function for patient interaction node
 */
function routePatientInteraction(state: MedicalDiagnosticStateType): string {
  // If we have pending questions, wait for user input (interrupt)
  if (state.pendingQuestions.length > 0) {
    return '__end__'; // This creates an interrupt
  }
  
  // If we just finished processing user response, continue with medical debate
  return 'medical_debate';
}

/**
 * Routing function for after user responds
 */
function routeAfterUserResponse(state: MedicalDiagnosticStateType): string {
  // After processing user response, we should either:
  // 1. Ask more questions if needed
  if (patientQuestionAgent.shouldGenerateQuestions(state)) {
    return 'patient_interaction';
  }
  
  // 2. Continue with medical debate to analyze the new information
  return 'medical_debate';
}

/**
 * Routing function after initialization
 */
function routeAfterInitialization(state: MedicalDiagnosticStateType): string {
  console.log("🔀 Routing after initialization");
  console.log("Has patient ID:", !!state.availableCaseInfo?.patientId);
  console.log("Current phase:", state.currentPhase);
  console.log("Awaiting user input:", state.awaitingUserInput);
  
  // If this was a resume from interrupt (case already initialized), 
  // check what the appropriate next step should be
  if (state.currentPhase === 'patient_interaction' || state.awaitingUserInput) {
    console.log("🔄 Resuming patient interaction flow");
    return 'patient_interaction';
  }
  
  if (state.currentPhase === 'final_diagnosis' || state.readyForDiagnosis) {
    console.log("🎯 Going to final assessment");
    return 'final_assessment';
  }
  
  // For new cases or normal flow, go to medical debate
  console.log("🏥 Starting medical debate");
  return 'medical_debate';
}

/**
 * Helper function to extract case information from text
 */
async function extractCaseInformation(
  caseText: string,
  config: RunnableConfig
): Promise<any> {
  console.log("🔍 Starting case information extraction...");
  
  if (!caseText || caseText.trim().length === 0) {
    console.log("⚠️  Empty case text, returning fallback");
    return {
      patientId: `CASE_${Date.now()}`,
      demographics: { age: 45, gender: 'unknown' },
      chiefComplaint: "No case information provided",
      historyOfPresentIllness: "Initial case presentation"
    };
  }
  
  try {
    const configuration = ensureMedicalConfiguration(config);
    const model = await loadChatModel(configuration.model);
    
    const extractionPrompt = `
Extract structured case information from this medical case and respond with ONLY valid JSON:

${caseText}

Required JSON format:
{
  "patientId": "generate a unique ID like CASE_123",
  "demographics": {"age": number, "gender": "male/female/unknown"},
  "chiefComplaint": "primary reason for visit",
  "historyOfPresentIllness": "detailed history",
  "pastMedicalHistory": [],
  "medications": [],
  "allergies": [],
  "familyHistory": [],
  "socialHistory": ""
}

If information is not available, use appropriate defaults. Respond with ONLY the JSON object.
`;

    console.log("🤖 Making LLM call for case extraction...");
    const response = await model.invoke([new HumanMessage(extractionPrompt)]);
    console.log("✅ LLM response received");
    
    try {
      const content = typeof response.content === 'string' ? response.content : '{}';
      const parsed = JSON.parse(content);
      
      // Ensure required fields exist
      const result = {
        patientId: parsed.patientId || `CASE_${Date.now()}`,
        demographics: parsed.demographics || { age: 45, gender: 'unknown' },
        chiefComplaint: parsed.chiefComplaint || caseText.substring(0, 100),
        historyOfPresentIllness: parsed.historyOfPresentIllness || caseText,
        pastMedicalHistory: Array.isArray(parsed.pastMedicalHistory) ? parsed.pastMedicalHistory : [],
        medications: Array.isArray(parsed.medications) ? parsed.medications : [],
        allergies: Array.isArray(parsed.allergies) ? parsed.allergies : [],
        familyHistory: Array.isArray(parsed.familyHistory) ? parsed.familyHistory : [],
        socialHistory: parsed.socialHistory || ""
      };
      
      console.log("✅ Case info parsed successfully:", result.patientId);
      return result;
    } catch (parseError) {
      console.log("⚠️  JSON parse error, using fallback:", parseError);
      // Fallback to basic case information
      return {
        patientId: `CASE_${Date.now()}`,
        demographics: { age: 45, gender: 'unknown' },
        chiefComplaint: caseText.substring(0, 100),
        historyOfPresentIllness: caseText,
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      };
    }
  } catch (llmError) {
    console.error("❌ LLM call failed:", llmError);
    // Fallback on LLM error
    return {
      patientId: `CASE_${Date.now()}`,
      demographics: { age: 45, gender: 'unknown' },
      chiefComplaint: caseText.substring(0, 100),
      historyOfPresentIllness: caseText,
      pastMedicalHistory: [],
      medications: [],
      allergies: [],
      familyHistory: [],
      socialHistory: ""
    };
  }
}

/**
 * Create and compile the medical diagnostic workflow graph
 * 
 * Flow: __start__ → initialize_case → [diagnostic_loop] → final_assessment → __end__
 * Where diagnostic_loop = medical_debate ↔ patient_interaction ↔ diagnostic_tools
 */
export function createMedicalGraph() {
  const workflow = new StateGraph(MedicalDiagnosticAnnotation, ConfigurationSchema)
    // Define all nodes
    .addNode("initialize_case", initializeCase)
    .addNode("medical_debate", medicalDebate)
    .addNode("patient_interaction", patientInteraction)
    .addNode("process_user_response", processUserResponse)
    .addNode("diagnostic_tools", executeDiagnosticTools)
    .addNode("final_assessment", finalAssessment)
    
    // ENTRY POINT: Initialize case once
    .addEdge("__start__", "initialize_case")
    
    // INITIALIZATION: Route appropriately after initialization
    .addConditionalEdges("initialize_case", routeAfterInitialization)
    
    // MAIN DIAGNOSTIC LOOP: Medical debate routes to different nodes
    .addConditionalEdges("medical_debate", routeWorkflow)
    
    // PATIENT INTERACTION: Can interrupt or continue debate
    .addConditionalEdges("patient_interaction", routePatientInteraction)
    
    // USER RESPONSE PROCESSING: Routes back to debate or more questions
    .addConditionalEdges("process_user_response", routeAfterUserResponse)
    
    // DIAGNOSTIC TOOLS: Always return to medical debate
    .addEdge("diagnostic_tools", "medical_debate")
    
    // FINAL ASSESSMENT: End the workflow
    .addEdge("final_assessment", "__end__");

  return workflow.compile({
    interruptBefore: ["patient_interaction"],
    interruptAfter: []
  });
}

// Export the compiled graph
export const medicalGraph = createMedicalGraph();