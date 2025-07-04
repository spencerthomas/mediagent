/**
 * Minimal Medical Diagnostic Graph
 * Simplified for LangGraph Studio compatibility
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { 
  MedicalDiagnosticAnnotation, 
  MedicalDiagnosticStateType
} from "./medical_state.js";
import { loadChatModel } from "../react_agent/utils.js";
import { ConfigurationSchema } from "../react_agent/configuration.js";
import { ensureMedicalConfiguration } from "./configuration.js";

/**
 * Node: Start Consultation
 * Initializes the medical consultation from user input
 */
async function startConsultation(
  state: MedicalDiagnosticStateType,
  _config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🏥 Starting medical consultation");
  
  // Get the user's input message
  const userMessage = state.messages[state.messages.length - 1];
  const caseDescription = typeof userMessage?.content === 'string' ? userMessage.content : "";
  
  console.log("📝 Case description:", caseDescription.substring(0, 100));
  
  // Initialize case info
  const caseInfo = {
    patientId: `CASE_${Date.now()}`,
    demographics: { age: 0, gender: 'unknown' },
    chiefComplaint: caseDescription.substring(0, 200),
    historyOfPresentIllness: caseDescription,
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
    confidenceLevel: 0.3,
    readyForDiagnosis: false,
    awaitingUserInput: true, // Set to true to trigger question asking
    pendingQuestions: [],
    differentialDiagnoses: [],
    messages: [
      ...state.messages,
      new AIMessage("Thank you for providing your case information. I need to ask you some follow-up questions to help with the diagnosis.")
    ]
  };
}

/**
 * Node: Ask Patient Questions
 * Generates and asks follow-up questions
 */
async function askPatientQuestions(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("❓ Asking patient questions");
  
  // Simple hardcoded questions for testing
  const basicQuestions = [
    "What is your age and gender?",
    "When did your symptoms start?",
    "Can you describe the severity of your symptoms on a scale of 1-10?",
    "Are you currently taking any medications?",
    "Do you have any known allergies or medical conditions?"
  ];
  
  // Generate questions based on round
  const roundNumber = state.interactionRound + 1;
  const questionText = roundNumber <= basicQuestions.length 
    ? basicQuestions[roundNumber - 1]
    : "Is there anything else you'd like to tell me about your symptoms?";
  
  return {
    interactionRound: roundNumber,
    awaitingUserInput: true,
    currentPhase: 'patient_interaction',
    messages: [
      ...state.messages,
      new AIMessage(`**Question ${roundNumber}:** ${questionText}`)
    ]
  };
}

/**
 * Node: Process Patient Answer
 * Processes the patient's response to questions
 */
async function processPatientAnswer(
  state: MedicalDiagnosticStateType,
  _config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("💬 Processing patient answer");
  
  // Get the latest user response
  const lastMessage = state.messages[state.messages.length - 1];
  const userResponse = typeof lastMessage?.content === 'string' ? lastMessage.content : "";
  
  console.log("📝 Patient response:", userResponse.substring(0, 100));
  
  // Update case info with the response
  const updatedHistory = `${state.availableCaseInfo.historyOfPresentIllness}\n\nQ${state.interactionRound}: ${userResponse}`;
  
  const updatedCaseInfo = {
    ...state.availableCaseInfo,
    historyOfPresentIllness: updatedHistory
  };
  
  // Determine if we need more questions or can proceed to analysis
  const needsMoreQuestions = state.interactionRound < 3;
  
  if (needsMoreQuestions) {
    return {
      availableCaseInfo: updatedCaseInfo,
      awaitingUserInput: true, // Still need more questions
      currentPhase: 'information_gathering',
      messages: [
        ...state.messages,
        new AIMessage("Thank you for that information. I have another question for you.")
      ]
    };
  } else {
    return {
      availableCaseInfo: updatedCaseInfo,
      awaitingUserInput: false,
      currentPhase: 'deliberation',
      readyForDiagnosis: true,
      messages: [
        ...state.messages,
        new AIMessage("Thank you for all the information. Let me analyze your case and provide a diagnostic assessment.")
      ]
    };
  }
}

/**
 * Node: Provide Diagnosis
 * Generates final diagnostic assessment
 */
async function provideDiagnosis(
  state: MedicalDiagnosticStateType,
  config: RunnableConfig
): Promise<Partial<MedicalDiagnosticStateType>> {
  console.log("🎯 Providing diagnosis");
  
  try {
    const configuration = ensureMedicalConfiguration(config);
    const model = await loadChatModel(configuration.model);
    
    const diagnosisPrompt = `
Based on the following patient information, provide a medical diagnostic assessment:

${state.availableCaseInfo.historyOfPresentIllness}

Please provide:
1. Most likely diagnosis
2. Key symptoms that support this diagnosis
3. Recommended next steps
4. Any red flags or concerns

Keep the response professional and helpful.
`;

    const response = await model.invoke([new HumanMessage(diagnosisPrompt)]);
    const diagnosticAssessment = typeof response.content === 'string' ? response.content : "Diagnostic assessment completed";
    
    return {
      currentPhase: 'final_diagnosis',
      readyForDiagnosis: true,
      confidenceLevel: 0.8,
      messages: [
        ...state.messages,
        new AIMessage(`## Medical Diagnostic Assessment\n\n${diagnosticAssessment}`)
      ]
    };
  } catch (error) {
    console.error("❌ Error generating diagnosis:", error);
    
    return {
      currentPhase: 'final_diagnosis',
      readyForDiagnosis: true,
      messages: [
        ...state.messages,
        new AIMessage("I've completed my assessment based on the information provided. Please consult with a healthcare professional for proper medical evaluation.")
      ]
    };
  }
}

/**
 * Routing function
 */
function routeConversation(state: MedicalDiagnosticStateType): string {
  // If ready for diagnosis, go to final step
  if (state.readyForDiagnosis || state.currentPhase === 'final_diagnosis') {
    return "provide_diagnosis";
  }
  
  // If awaiting user input, ask questions
  if (state.awaitingUserInput) {
    return "ask_questions";
  }
  
  // Otherwise process the patient's answer
  return "process_answer";
}

/**
 * Create the minimal medical graph
 */
export function createMinimalMedicalGraph() {
  const workflow = new StateGraph(MedicalDiagnosticAnnotation, ConfigurationSchema)
    .addNode("start_consultation", startConsultation)
    .addNode("ask_questions", askPatientQuestions)
    .addNode("process_answer", processPatientAnswer)
    .addNode("provide_diagnosis", provideDiagnosis)
    
    // Entry point
    .addEdge("__start__", "start_consultation")
    
    // Main conversation flow
    .addConditionalEdges("start_consultation", routeConversation)
    .addConditionalEdges("ask_questions", () => "__end__") // This creates the interrupt for user input
    .addConditionalEdges("process_answer", routeConversation)
    .addEdge("provide_diagnosis", "__end__");

  return workflow.compile({
    interruptBefore: [],
    interruptAfter: ["ask_questions"] // Interrupt after asking questions
  });
}

// Export the minimal graph
export const minimalMedicalGraph = createMinimalMedicalGraph();