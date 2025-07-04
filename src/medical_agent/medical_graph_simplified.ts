/**
 * Simplified Medical Diagnostic Agent Graph
 * Focused on clean patient interaction and reliable state management
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { 
  MedicalDiagnosticAnnotation, 
  MedicalDiagnosticStateType, 
  DiagnosticPhase 
} from "./medical_state.js";
import { ChainOfDebate } from "./chain_of_debate.js";
import { PatientQuestionAgent } from "./patient_question_agent.js";
import { loadChatModel } from "../react_agent/utils.js";
import { ConfigurationSchema } from "../react_agent/configuration.js";
import { ensureMedicalConfiguration } from "./configuration.js";

// Initialize components
const patientQuestionAgent = new PatientQuestionAgent(loadChatModel);
const chainOfDebate = new ChainOfDebate(loadChatModel);

/**
 * Node: Initialize Case (ONLY ONCE)
 * Sets up the initial case information from the first user message
 */
async function initializeCase(
  state: MedicalDiagnosticStateType,
  _config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🚨 initializeCase - Entry point");
  
  // Get the first message for case initialization
  const firstMessage = state.messages[0];
  const caseText = typeof firstMessage?.content === 'string' ? firstMessage.content : "Patient case";
  
  console.log("📝 Initializing with case:", caseText.substring(0, 100));
  
  // Simple case info extraction without LLM call
  const caseInfo = {
    patientId: `CASE_${Date.now()}`,
    demographics: { age: 0, gender: 'unknown' },
    chiefComplaint: caseText.substring(0, 200),
    historyOfPresentIllness: caseText,
    pastMedicalHistory: [],
    medications: [],
    allergies: [],
    familyHistory: [],
    socialHistory: ""
  };
  
  return {
    availableCaseInfo: caseInfo,
    currentPhase: 'initial_assessment',
    costBudget: 1000,
    interactionRound: 0,
    confidenceLevel: 0.2,
    readyForDiagnosis: false,
    awaitingUserInput: false,
    pendingQuestions: [],
    messages: [
      ...state.messages,
      new AIMessage(`I've received your case information. Let me begin the diagnostic process and ask you some follow-up questions.`)
    ]
  };
}

/**
 * Node: Generate Patient Questions
 * Creates questions for the patient based on diagnostic needs
 */
async function generatePatientQuestions(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("❓ Generating patient questions");
  
  // Check if we should generate questions
  if (!patientQuestionAgent.shouldGenerateQuestions(state)) {
    console.log("✅ No more questions needed, proceeding to medical analysis");
    return {
      currentPhase: 'deliberation',
      awaitingUserInput: false
    };
  }
  
  try {
    const questions = await patientQuestionAgent.generateQuestions(state, config);
    
    if (questions.length > 0) {
      const roundNumber = state.interactionRound + 1;
      console.log(`📋 Generated ${questions.length} questions for round ${roundNumber}`);
      
      return {
        pendingQuestions: questions,
        questionHistory: [...state.questionHistory, ...questions],
        awaitingUserInput: true,
        interactionRound: roundNumber,
        currentPhase: 'patient_interaction',
        messages: [
          ...state.messages,
          new AIMessage(`**Patient Interview - Round ${roundNumber}**\n\nI need some additional information to help with your diagnosis:\n\n${
            questions.map((q, i) => `${i + 1}. ${q.question}`).join('\n')
          }\n\nPlease provide your responses.`)
        ]
      };
    }
  } catch (error) {
    console.error("❌ Error generating questions:", error);
  }
  
  // If no questions generated or error, proceed to medical analysis
  return {
    currentPhase: 'deliberation',
    awaitingUserInput: false
  };
}

/**
 * Node: Process Patient Response
 * Handles user responses to medical questions
 */
async function processPatientResponse(
  state: MedicalDiagnosticStateType,
  _config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("💬 Processing patient response");
  
  // Get the latest user message
  const lastMessage = state.messages[state.messages.length - 1];
  const userResponse = typeof lastMessage?.content === 'string' ? lastMessage.content : "";
  
  console.log("📝 User response:", userResponse.substring(0, 100));
  
  // Simple response processing - update case info with user response
  const updatedCaseInfo = {
    ...state.availableCaseInfo,
    historyOfPresentIllness: `${state.availableCaseInfo.historyOfPresentIllness}\n\nPatient Response Round ${state.interactionRound}: ${userResponse}`
  };
  
  // Clear pending questions and continue
  return {
    availableCaseInfo: updatedCaseInfo,
    pendingQuestions: [],
    awaitingUserInput: false,
    currentPhase: 'information_gathering',
    requiredInformationGathered: [...state.requiredInformationGathered, `round_${state.interactionRound}_response`],
    messages: [
      ...state.messages,
      new AIMessage(`Thank you for the additional information. Let me analyze this with the medical team.`)
    ]
  };
}

/**
 * Node: Medical Analysis
 * Runs the chain of debate for diagnostic analysis
 */
async function medicalAnalysis(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🏥 Starting medical analysis");
  
  try {
    const analysisResults = await chainOfDebate.orchestrateDebate(state, config);
    console.log("✅ Medical analysis completed");
    
    // Determine if we need more patient interaction or can proceed to final diagnosis
    const shouldAskMoreQuestions = 
      state.interactionRound < 3 && 
      (analysisResults.confidenceLevel || 0) < 0.7 &&
      patientQuestionAgent.shouldGenerateQuestions({...state, ...analysisResults});
    
    if (shouldAskMoreQuestions) {
      return {
        ...analysisResults,
        currentPhase: 'information_gathering'
      };
    } else {
      return {
        ...analysisResults,
        currentPhase: 'final_diagnosis',
        readyForDiagnosis: true
      };
    }
  } catch (error) {
    console.error("❌ Error in medical analysis:", error);
    
    // Fallback: proceed to final diagnosis with current information
    return {
      currentPhase: 'final_diagnosis',
      readyForDiagnosis: true,
      confidenceLevel: 0.5,
      messages: [
        ...state.messages,
        new AIMessage("Medical analysis completed. Proceeding to final diagnosis based on available information.")
      ]
    };
  }
}

/**
 * Node: Final Diagnosis
 * Generates the final diagnostic assessment
 */
async function finalDiagnosis(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🎯 Generating final diagnosis");
  
  try {
    const configuration = ensureMedicalConfiguration(config);
    const model = await loadChatModel(configuration.model);
    
    const topDiagnosis = state.differentialDiagnoses.length > 0 
      ? state.differentialDiagnoses[0] 
      : { condition: "Further evaluation needed", probability: 50, supportingEvidence: [], reasoning: "Insufficient information" };
    
    const diagnosisPrompt = `
Based on the medical case analysis, provide a final diagnostic assessment:

Patient: ${state.availableCaseInfo.chiefComplaint}
Top Diagnosis: ${topDiagnosis.condition} (${topDiagnosis.probability}%)
Confidence: ${(state.confidenceLevel * 100).toFixed(1)}%

Provide a concise final assessment including:
1. Primary diagnosis with reasoning
2. Key supporting evidence
3. Recommended next steps
4. Any important considerations
`;

    const response = await model.invoke([new HumanMessage(diagnosisPrompt)]);
    const finalAssessment = typeof response.content === 'string' ? response.content : "Assessment completed";
    
    return {
      finalDiagnosis: topDiagnosis,
      currentPhase: 'final_diagnosis',
      readyForDiagnosis: true,
      messages: [
        ...state.messages,
        new AIMessage(`## Final Diagnostic Assessment\n\n${finalAssessment}`)
      ]
    };
  } catch (error) {
    console.error("❌ Error generating final diagnosis:", error);
    
    return {
      currentPhase: 'final_diagnosis',
      readyForDiagnosis: true,
      messages: [
        ...state.messages,
        new AIMessage("Final diagnostic assessment completed based on available information.")
      ]
    };
  }
}

/**
 * Routing Functions
 */
function routeFromInitialization(_state: MedicalDiagnosticStateType): string {
  return "generate_questions";
}

function routeFromQuestions(state: MedicalDiagnosticStateType): string {
  if (state.awaitingUserInput && state.pendingQuestions.length > 0) {
    return "__end__"; // This creates the interrupt for user input
  }
  return "medical_analysis";
}

function routeFromResponse(state: MedicalDiagnosticStateType): string {
  // After processing response, check if we need more questions or can analyze
  if (state.currentPhase === 'information_gathering') {
    return "generate_questions";
  }
  return "medical_analysis";
}

function routeFromAnalysis(state: MedicalDiagnosticStateType): string {
  if (state.currentPhase === 'final_diagnosis' || state.readyForDiagnosis) {
    return "final_diagnosis";
  }
  if (state.currentPhase === 'information_gathering') {
    return "generate_questions";
  }
  return "final_diagnosis";
}

/**
 * Create the simplified medical diagnostic workflow graph
 */
export function createSimplifiedMedicalGraph() {
  const workflow = new StateGraph(MedicalDiagnosticAnnotation, ConfigurationSchema)
    // Define nodes
    .addNode("initialize_case", initializeCase)
    .addNode("generate_questions", generatePatientQuestions)
    .addNode("process_response", processPatientResponse)
    .addNode("medical_analysis", medicalAnalysis)
    .addNode("final_diagnosis", finalDiagnosis)
    
    // Entry point - initialize case ONCE
    .addEdge("__start__", "initialize_case")
    
    // Routing from each node
    .addConditionalEdges("initialize_case", routeFromInitialization)
    .addConditionalEdges("generate_questions", routeFromQuestions)
    .addConditionalEdges("process_response", routeFromResponse)
    .addConditionalEdges("medical_analysis", routeFromAnalysis)
    .addEdge("final_diagnosis", "__end__");

  return workflow.compile({
    interruptBefore: [], // No interrupts before nodes
    interruptAfter: ["generate_questions"] // Interrupt after generating questions, before user response
  });
}

// Export the simplified graph
export const simplifiedMedicalGraph = createSimplifiedMedicalGraph();