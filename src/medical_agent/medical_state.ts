/**
 * Medical Diagnostic State Management
 * Defines the state structure for the medical superintelligence agent
 */

import { BaseMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";

export interface DiagnosisHypothesis {
  condition: string;
  probability: number;
  supportingEvidence: string[];
  reasoning: string;
  icd10Code?: string;
}

export interface CaseInformation {
  patientId: string;
  demographics: {
    age: number;
    gender: string;
    occupation?: string;
  };
  chiefComplaint: string;
  historyOfPresentIllness?: string;
  pastMedicalHistory?: string[];
  medications?: string[];
  allergies?: string[];
  familyHistory?: string[];
  socialHistory?: string;
  reviewOfSystems?: Record<string, string>;
  physicalExam?: Record<string, string>;
}

export interface TestResult {
  testType: string;
  testName: string;
  result: string;
  cost: number;
  diagnosticValue: number;
  requestedBy: string;
  timestamp: Date;
}

export interface PatientQuestion {
  id: string;
  requestingAgent: 'hypothesis' | 'test_chooser' | 'challenger' | 'stewardship' | 'checklist';
  category: 'history' | 'symptoms' | 'examination' | 'family' | 'social' | 'medications' | 'allergies';
  question: string;
  priority: 'high' | 'medium' | 'low';
  timestamp: Date;
}

export interface PatientResponse {
  questionId: string;
  response: string;
  timestamp: Date;
}

export interface AgentTurn {
  agentRole: 'hypothesis' | 'test_chooser' | 'challenger' | 'stewardship' | 'checklist';
  reasoning: string;
  recommendations: string[];
  diagnosisUpdates?: DiagnosisHypothesis[];
  testsRequested?: string[];
  questionsRequested?: PatientQuestion[];
  costAnalysis?: {
    estimatedCost: number;
    costBenefit: string;
  };
  biasesIdentified?: string[];
}

export type DiagnosticPhase = 
  | 'case_presentation'
  | 'initial_assessment' 
  | 'information_gathering'
  | 'patient_interaction'
  | 'test_selection'
  | 'deliberation'
  | 'final_diagnosis';

export interface MedicalDiagnosticState {
  // Message history for conversation
  messages: BaseMessage[];
  
  // Medical diagnostic state
  differentialDiagnoses: DiagnosisHypothesis[];
  availableCaseInfo: CaseInformation;
  revealedInformation: string[];
  diagnosticTests: TestResult[];
  cumulativeCost: number;
  costBudget: number;
  
  // Workflow state
  currentPhase: DiagnosticPhase;
  debateRound: number;
  agentTurns: AgentTurn[];
  
  // Patient interaction state
  pendingQuestions: PatientQuestion[];
  awaitingUserInput: boolean;
  questionHistory: PatientQuestion[];
  userResponses: PatientResponse[];
  
  // Decision tracking
  confidenceLevel: number;
  readyForDiagnosis: boolean;
  finalDiagnosis?: DiagnosisHypothesis;
  
  // Quality metrics
  biasesDetected: string[];
  reasoningQuality: number;
}

// LangGraph State Annotation for Medical Diagnostic Agent
export const MedicalDiagnosticAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  differentialDiagnoses: Annotation<DiagnosisHypothesis[]>({
    reducer: (current, update) => update || current,
    default: () => [],
  }),
  availableCaseInfo: Annotation<CaseInformation>({
    reducer: (current, update) => ({ ...current, ...update }),
    default: () => ({
      patientId: '',
      demographics: { age: 0, gender: '' },
      chiefComplaint: '',
    }),
  }),
  revealedInformation: Annotation<string[]>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
  diagnosticTests: Annotation<TestResult[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  cumulativeCost: Annotation<number>({
    reducer: (current, update) => current + update,
    default: () => 0,
  }),
  costBudget: Annotation<number>({
    reducer: (current, update) => update || current,
    default: () => 1000, // Default budget of $1000
  }),
  currentPhase: Annotation<DiagnosticPhase>({
    reducer: (current, update) => update || current,
    default: () => 'case_presentation',
  }),
  debateRound: Annotation<number>({
    reducer: (current, update) => update || current,
    default: () => 0,
  }),
  agentTurns: Annotation<AgentTurn[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  confidenceLevel: Annotation<number>({
    reducer: (current, update) => update || current,
    default: () => 0,
  }),
  readyForDiagnosis: Annotation<boolean>({
    reducer: (current, update) => update !== undefined ? update : current,
    default: () => false,
  }),
  finalDiagnosis: Annotation<DiagnosisHypothesis | undefined>({
    reducer: (current, update) => update || current,
    default: () => undefined,
  }),
  biasesDetected: Annotation<string[]>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
  reasoningQuality: Annotation<number>({
    reducer: (current, update) => update || current,
    default: () => 0,
  }),
  pendingQuestions: Annotation<PatientQuestion[]>({
    reducer: (current, update) => update || current,
    default: () => [],
  }),
  awaitingUserInput: Annotation<boolean>({
    reducer: (current, update) => update !== undefined ? update : current,
    default: () => false,
  }),
  questionHistory: Annotation<PatientQuestion[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  userResponses: Annotation<PatientResponse[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
});

export type MedicalDiagnosticStateType = typeof MedicalDiagnosticAnnotation.State;