/**
 * Patient Question Agent
 * Generates targeted follow-up questions based on diagnostic needs
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { HumanMessage } from "@langchain/core/messages";
import { 
  MedicalDiagnosticStateType, 
  PatientQuestion
} from "./medical_state.js";
import { ensureMedicalConfiguration } from "./configuration.js";

export class PatientQuestionAgent {
  constructor(
    private loadChatModel: (modelName: string) => Promise<any>
  ) {}

  async generateQuestions(
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<PatientQuestion[]> {
    const configuration = ensureMedicalConfiguration(config);
    const model = await this.loadChatModel(configuration.model);

    const questionPrompt = this.buildQuestionPrompt(state);
    const response = await model.invoke([new HumanMessage(questionPrompt)]);
    
    const questions = this.parseQuestions(response.content as string, state);
    return this.prioritizeQuestions(questions, state);
  }

  private buildQuestionPrompt(state: MedicalDiagnosticStateType): string {
    const currentDiagnoses = state.differentialDiagnoses
      .slice(0, 5)
      .map(d => `${d.condition} (${d.probability}%)`)
      .join(', ');

    const availableInfo = this.summarizeAvailableInfo(state);
    const recentAgentRecommendations = this.getRecentRecommendations(state);
    const missingInfo = this.identifyMissingInformation(state);
    const roundStrategy = this.getRoundStrategy(state.interactionRound, state.differentialDiagnoses);

    return `
You are a medical information specialist conducting Round ${state.interactionRound + 1} of patient interview. Generate 3-5 targeted follow-up questions based on current diagnostic hypotheses and information gaps.

CURRENT DIAGNOSTIC STATE:
Chief Complaint: ${state.availableCaseInfo.chiefComplaint}
Top Differential Diagnoses: ${currentDiagnoses || 'Initial assessment in progress'}
Confidence Level: ${(state.confidenceLevel * 100).toFixed(1)}%
Interview Round: ${state.interactionRound + 1}
Current Phase: ${state.currentPhase}

AVAILABLE INFORMATION:
${availableInfo}

MISSING CRITICAL INFORMATION:
${missingInfo}

ROUND STRATEGY:
${roundStrategy}

RECENT AGENT RECOMMENDATIONS:
${recentAgentRecommendations}

DYNAMIC QUESTION GENERATION GUIDELINES:
1. **Discriminate between diagnoses**: Ask questions that help differentiate between the top 3-5 differential diagnoses
2. **Target information gaps**: Focus on the most critical missing information for this round
3. **Progressive refinement**: Build on previous responses to drill deeper into relevant areas
4. **Diagnostic specificity**: Questions should help increase or decrease probability of specific conditions
5. **Round-appropriate depth**: Early rounds = broad information, later rounds = specific discriminating details

Generate questions in this JSON format:
{
  "questions": [
    {
      "category": "symptoms|history|examination|family|social|medications|allergies",
      "question": "Clear, specific question text",
      "priority": "high|medium|low",
      "reasoning": "How this question discriminates between current differential diagnoses",
      "targets_diagnoses": ["diagnosis1", "diagnosis2"]
    }
  ]
}

Focus on questions that will most effectively narrow down the differential diagnosis. Maximum 5 questions.
`;
  }

  private identifyMissingInformation(state: MedicalDiagnosticStateType): string {
    const missing = [];
    const caseInfo = state.availableCaseInfo;
    const gathered = state.requiredInformationGathered;
    
    if (!caseInfo.demographics.age || caseInfo.demographics.age === 0) missing.push("Patient age");
    if (!caseInfo.demographics.gender || caseInfo.demographics.gender === 'unknown') missing.push("Patient gender");
    if (!gathered.includes('medications')) missing.push("Current medications");
    if (!gathered.includes('allergies')) missing.push("Known allergies");
    if (!gathered.includes('family_history')) missing.push("Family medical history");
    if (!gathered.includes('symptom_timeline')) missing.push("Symptom onset and timeline");
    if (!gathered.includes('symptom_severity')) missing.push("Symptom severity and characteristics");
    if (!caseInfo.pastMedicalHistory || !Array.isArray(caseInfo.pastMedicalHistory) || caseInfo.pastMedicalHistory.length === 0) missing.push("Past medical history");
    if (!caseInfo.socialHistory) missing.push("Social history (smoking, alcohol, occupation)");
    
    return missing.length > 0 ? missing.join(', ') : 'All basic information collected';
  }

  private getRoundStrategy(round: number, diagnoses: any[]): string {
    switch (round) {
      case 0:
        return "ROUND 1 FOCUS: Establish basic demographics, symptom timeline, and severity. Gather foundational information for initial differential diagnosis.";
      case 1:
        return "ROUND 2 FOCUS: Collect medical history, medications, and allergies. Begin exploring specific symptom characteristics that differentiate between initial diagnostic hypotheses.";
      case 2:
        return "ROUND 3 FOCUS: Target questions to discriminate between top differential diagnoses. Ask about associated symptoms, risk factors, and family history relevant to leading conditions.";
      default:
        if (diagnoses.length > 2) {
          return `ROUND ${round + 1} FOCUS: Precision targeting - ask highly specific questions to definitively rule in/out the top ${Math.min(diagnoses.length, 3)} competing diagnoses.`;
        }
        return `ROUND ${round + 1} FOCUS: Final clarification - address any remaining diagnostic uncertainty with targeted questions about the most likely condition(s).`;
    }
  }

  private summarizeAvailableInfo(state: MedicalDiagnosticStateType): string {
    const info = state.availableCaseInfo;
    const summary = [
      `Age: ${info.demographics.age}, Gender: ${info.demographics.gender}`
    ];

    if (info.historyOfPresentIllness) {
      summary.push(`History: ${info.historyOfPresentIllness.substring(0, 200)}...`);
    }

    if (info.pastMedicalHistory?.length && Array.isArray(info.pastMedicalHistory)) {
      summary.push(`Past Medical History: ${info.pastMedicalHistory.join(', ')}`);
    }

    if (info.medications?.length && Array.isArray(info.medications)) {
      summary.push(`Medications: ${info.medications.join(', ')}`);
    }

    if (info.familyHistory?.length && Array.isArray(info.familyHistory)) {
      summary.push(`Family History: ${info.familyHistory.join(', ')}`);
    }

    if (state.diagnosticTests.length > 0) {
      summary.push(`Tests: ${state.diagnosticTests.map(t => `${t.testName}: ${t.result}`).join(', ')}`);
    }

    return summary.join('\n');
  }

  private getRecentRecommendations(state: MedicalDiagnosticStateType): string {
    const recentTurns = state.agentTurns.slice(-3);
    return recentTurns
      .map(turn => `${turn.agentRole}: ${turn.recommendations.join('; ')}`)
      .join('\n');
  }

  private parseQuestions(content: string, state: MedicalDiagnosticStateType): PatientQuestion[] {
    try {
      const parsed = JSON.parse(content);
      
      return parsed.questions.map((q: any, index: number) => ({
        id: `q_${Date.now()}_${index}`,
        requestingAgent: 'hypothesis' as const,
        category: q.category,
        question: q.question,
        priority: q.priority,
        timestamp: new Date()
      }));
    } catch (error) {
      // Fallback: generate default questions based on current state
      return this.generateDefaultQuestions(state);
    }
  }

  private generateDefaultQuestions(state: MedicalDiagnosticStateType): PatientQuestion[] {
    const questions: PatientQuestion[] = [];
    
    // Always ask about symptom timeline if not clear
    if (!state.availableCaseInfo.historyOfPresentIllness?.includes('started') && 
        !state.availableCaseInfo.historyOfPresentIllness?.includes('began')) {
      questions.push({
        id: `q_${Date.now()}_timeline`,
        requestingAgent: 'hypothesis',
        category: 'symptoms',
        question: 'When did your symptoms first begin, and how have they changed over time?',
        priority: 'high',
        timestamp: new Date()
      });
    }

    // Ask about pain characteristics if pain is mentioned
    if (state.availableCaseInfo.chiefComplaint.toLowerCase().includes('pain')) {
      questions.push({
        id: `q_${Date.now()}_pain`,
        requestingAgent: 'hypothesis',
        category: 'symptoms',
        question: 'On a scale of 1-10, how would you rate your pain, and what makes it better or worse?',
        priority: 'high',
        timestamp: new Date()
      });
    }

    // Ask about medications if none listed
    if (!state.availableCaseInfo.medications?.length) {
      questions.push({
        id: `q_${Date.now()}_meds`,
        requestingAgent: 'hypothesis',
        category: 'medications',
        question: 'Are you currently taking any medications, including over-the-counter drugs or supplements?',
        priority: 'medium',
        timestamp: new Date()
      });
    }

    return questions;
  }

  private prioritizeQuestions(questions: PatientQuestion[], state: MedicalDiagnosticStateType): PatientQuestion[] {
    // Score questions based on diagnostic value
    const scoredQuestions = questions.map(q => ({
      ...q,
      score: this.calculateQuestionScore(q, state)
    }));

    // Sort by score (descending) and return top 5
    return scoredQuestions
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(({ score, ...question }) => question);
  }

  private calculateQuestionScore(question: PatientQuestion, state: MedicalDiagnosticStateType): number {
    let score = 0;

    // Priority weighting
    switch (question.priority) {
      case 'high': score += 3; break;
      case 'medium': score += 2; break;
      case 'low': score += 1; break;
    }

    // Category weighting based on current phase
    switch (state.currentPhase) {
      case 'initial_assessment':
        if (question.category === 'symptoms' || question.category === 'history') score += 2;
        break;
      case 'information_gathering':
        if (question.category === 'family' || question.category === 'social') score += 2;
        break;
      case 'patient_interaction':
        score += 1; // All questions valuable in this phase
        break;
    }

    // Boost score if question relates to high-probability diagnoses
    if (state.differentialDiagnoses.length > 0) {
      const topDiagnosis = state.differentialDiagnoses[0];
      if (topDiagnosis.probability > 60) {
        score += 2;
      }
    }

    return score;
  }

  shouldGenerateQuestions(state: MedicalDiagnosticStateType): boolean {
    // Don't generate questions if we already have pending ones
    if (state.pendingQuestions.length > 0) return false;
    
    // MANDATORY: Ensure minimum 3 rounds of patient interaction
    if (state.interactionRound < 3) return true;
    
    // Check if we need more information (lowered threshold)
    if (state.confidenceLevel < 0.4) return true;
    
    // Check if we have too many competing diagnoses
    if (state.differentialDiagnoses.length > 4) return true;
    
    // Check for missing essential information
    if (this.hasMissingEssentialInfo(state)) return true;
    
    // Check if agents are requesting information
    const recentTurns = state.agentTurns.slice(-2);
    const hasInfoRequests = recentTurns.some(turn => 
      turn.recommendations.some(rec => 
        rec.toLowerCase().includes('need') || 
        rec.toLowerCase().includes('require') ||
        rec.toLowerCase().includes('ask') ||
        rec.toLowerCase().includes('clarify')
      )
    );
    
    if (hasInfoRequests) return true;
    
    // Check if we're in information gathering phase
    if (state.currentPhase === 'information_gathering') return true;
    
    // Additional round if differential diagnoses need refinement
    if (state.interactionRound < 5 && state.differentialDiagnoses.length > 2) return true;
    
    return false;
  }

  private hasMissingEssentialInfo(state: MedicalDiagnosticStateType): boolean {
    const caseInfo = state.availableCaseInfo;
    const gathered = state.requiredInformationGathered;
    
    // Check for missing demographic information
    if (!caseInfo.demographics.age || caseInfo.demographics.age === 0) return true;
    if (!caseInfo.demographics.gender || caseInfo.demographics.gender === 'unknown') return true;
    
    // Check for missing medical history
    if (!gathered.includes('medications') && (!caseInfo.medications || !Array.isArray(caseInfo.medications) || caseInfo.medications.length === 0)) return true;
    if (!gathered.includes('allergies') && (!caseInfo.allergies || !Array.isArray(caseInfo.allergies) || caseInfo.allergies.length === 0)) return true;
    if (!gathered.includes('family_history') && (!caseInfo.familyHistory || !Array.isArray(caseInfo.familyHistory) || caseInfo.familyHistory.length === 0)) return true;
    
    // Check for incomplete symptom description
    if (!gathered.includes('symptom_timeline') && 
        (!caseInfo.historyOfPresentIllness || 
         !caseInfo.historyOfPresentIllness.toLowerCase().includes('started') &&
         !caseInfo.historyOfPresentIllness.toLowerCase().includes('began'))) return true;
    
    // Check for missing symptom severity/characteristics
    if (state.interactionRound >= 2 && !gathered.includes('symptom_severity')) return true;
    
    return false;
  }
}