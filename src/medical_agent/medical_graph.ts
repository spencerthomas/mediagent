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
  // Initialize with basic case presentation
  const lastMessage = state.messages[state.messages.length - 1];
  const caseText = typeof lastMessage?.content === 'string' ? lastMessage.content : "";
  
  // Extract basic case information from the input
  const caseInfo = await extractCaseInformation(caseText, config);
  
  return {
    availableCaseInfo: caseInfo,
    currentPhase: 'initial_assessment',
    costBudget: 1000, // Default budget
    messages: [
      ...state.messages,
      new AIMessage(`Case initialized: ${caseInfo.patientId} - ${caseInfo.chiefComplaint}`)
    ]
  };
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
      return {
        pendingQuestions: questions,
        questionHistory: [...state.questionHistory, ...questions],
        awaitingUserInput: true,
        currentPhase: 'patient_interaction',
        messages: [
          ...state.messages,
          new AIMessage(`I need some additional information to help with your diagnosis. Please answer the following questions:\n\n${
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
    
    return {
      availableCaseInfo: updatedCaseInfo,
      pendingQuestions: [], // Clear pending questions
      awaitingUserInput: false,
      currentPhase: 'information_gathering', // Continue with analysis
      messages: [
        ...state.messages,
        new AIMessage("Thank you for the additional information. Let me analyze this with my medical team.")
      ]
    };
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
 */
function routeWorkflow(state: MedicalDiagnosticStateType): string {
  // Check if awaiting user input
  if (state.awaitingUserInput) {
    return 'patient_interaction';
  }
  
  // Check if final diagnosis is ready
  if (state.readyForDiagnosis || state.currentPhase === 'final_diagnosis') {
    return 'final_assessment';
  }
  
  // Check if over budget
  if (state.cumulativeCost >= state.costBudget) {
    return 'final_assessment';
  }
  
  // Check if maximum rounds reached
  if (state.debateRound >= 5) {
    return 'final_assessment';
  }
  
  // Check if patient interaction is needed
  if (patientQuestionAgent.shouldGenerateQuestions(state)) {
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
  // If we have pending questions, wait for user input
  if (state.pendingQuestions.length > 0) {
    return '__end__'; // This creates an interrupt
  }
  
  // Otherwise continue with medical debate
  return 'medical_debate';
}

/**
 * Helper function to extract case information from text
 */
async function extractCaseInformation(
  caseText: string,
  config: RunnableConfig
): Promise<any> {
  const configuration = ensureMedicalConfiguration(config);
  const model = await loadChatModel(configuration.model);
  
  const extractionPrompt = `
Extract structured case information from this medical case:

${caseText}

Extract and format as JSON:
- patientId: generate a unique ID
- demographics: age, gender, occupation if mentioned
- chiefComplaint: primary reason for visit
- historyOfPresentIllness: detailed history if available
- pastMedicalHistory: relevant past medical history
- medications: current medications
- allergies: known allergies
- familyHistory: relevant family history
- socialHistory: relevant social history

If information is not available, use appropriate defaults or null values.
`;

  const response = await model.invoke([new HumanMessage(extractionPrompt)]);
  
  try {
    const content = typeof response.content === 'string' ? response.content : '{}';
    return JSON.parse(content);
  } catch (error) {
    // Fallback to basic case information
    return {
      patientId: `CASE_${Date.now()}`,
      demographics: {
        age: 45,
        gender: 'unknown'
      },
      chiefComplaint: caseText.substring(0, 100) + '...',
      historyOfPresentIllness: caseText
    };
  }
}

/**
 * Create and compile the medical diagnostic workflow graph
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
    
    // Set entry point
    .addEdge("__start__", "initialize_case")
    
    // Define workflow routing
    .addConditionalEdges(
      "initialize_case",
      () => "medical_debate"
    )
    .addConditionalEdges(
      "medical_debate",
      routeWorkflow
    )
    .addConditionalEdges(
      "patient_interaction",
      routePatientInteraction
    )
    .addConditionalEdges(
      "process_user_response",
      () => "medical_debate"
    )
    .addConditionalEdges(
      "diagnostic_tools",
      () => "medical_debate"
    )
    .addEdge("final_assessment", "__end__");

  return workflow.compile({
    interruptBefore: ["patient_interaction"],
    interruptAfter: []
  });
}

// Export the compiled graph
export const medicalGraph = createMedicalGraph();