/**
 * Gatekeeper Information Control System
 * Manages selective revelation of case information to simulate real-world diagnostic workflow
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { HumanMessage } from "@langchain/core/messages";
import { MedicalDiagnosticStateType, CaseInformation } from "./medical_state.js";
import { ensureMedicalConfiguration } from "./configuration.js";

export interface InformationRequest {
  requestedBy: string;
  requestType: 'history' | 'physical_exam' | 'lab_results' | 'imaging' | 'specialist_consultation';
  specificRequest: string;
  justification: string;
  estimatedCost: number;
}

export interface InformationResponse {
  granted: boolean;
  information: string;
  cost: number;
  additionalContext?: string;
  followUpQuestions?: string[];
}

export class CaseInformationGatekeeper {
  private revealedSections: Set<string> = new Set();
  private informationHierarchy: Map<string, number> = new Map();

  constructor(
    private loadChatModel: (modelName: string) => Promise<any>
  ) {
    // Define information hierarchy (lower number = easier to obtain)
    this.informationHierarchy.set('demographics', 0);
    this.informationHierarchy.set('chief_complaint', 0);
    this.informationHierarchy.set('history_present_illness', 1);
    this.informationHierarchy.set('past_medical_history', 1);
    this.informationHierarchy.set('medications', 1);
    this.informationHierarchy.set('allergies', 1);
    this.informationHierarchy.set('social_history', 2);
    this.informationHierarchy.set('family_history', 2);
    this.informationHierarchy.set('review_of_systems', 2);
    this.informationHierarchy.set('physical_exam', 3);
    this.informationHierarchy.set('lab_results', 4);
    this.informationHierarchy.set('imaging', 5);
    this.informationHierarchy.set('specialist_notes', 6);
  }

  async processInformationRequest(
    request: InformationRequest,
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<InformationResponse> {
    // Evaluate request validity
    const isValidRequest = await this.validateRequest(request, state, config);
    if (!isValidRequest) {
      return {
        granted: false,
        information: "Request not specific enough or not clinically justified.",
        cost: 0
      };
    }

    // Check budget constraints
    if (state.cumulativeCost + request.estimatedCost > state.costBudget) {
      return {
        granted: false,
        information: "Request exceeds remaining budget.",
        cost: 0,
        additionalContext: `Remaining budget: $${state.costBudget - state.cumulativeCost}`
      };
    }

    // Generate response based on request type
    const response = await this.generateInformationResponse(request, state, config);
    
    // Track revealed information
    this.revealedSections.add(request.requestType);
    
    return response;
  }

  private async validateRequest(
    request: InformationRequest,
    state: MedicalDiagnosticStateType,
    _config: RunnableConfig
  ): Promise<boolean> {
    // Use AI to validate clinical relevance
    const model = await this.loadChatModel("gpt-4");
    
    const validationPrompt = `
You are a medical information gatekeeper. Evaluate if this information request is clinically justified:

Request: ${request.specificRequest}
Justification: ${request.justification}
Requested by: ${request.requestedBy}

Current case context:
- Chief complaint: ${state.availableCaseInfo.chiefComplaint}
- Current diagnoses under consideration: ${state.differentialDiagnoses.map(d => d.condition).join(', ')}
- Previous information revealed: ${state.revealedInformation.join(', ')}

Is this request:
1. Clinically relevant and justified?
2. Specific enough to be actionable?
3. Appropriately sequenced (not requesting advanced tests before basics)?

Respond with YES or NO and brief reasoning.
`;

    const response = await model.invoke([new HumanMessage(validationPrompt)]);
    return response.content.toUpperCase().includes('YES');
  }

  private async generateInformationResponse(
    request: InformationRequest,
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<InformationResponse> {
    const caseInfo = state.availableCaseInfo;
    
    switch (request.requestType) {
      case 'history':
        return this.generateHistoryResponse(request, caseInfo);
      
      case 'physical_exam':
        return this.generatePhysicalExamResponse(request, caseInfo, config);
      
      case 'lab_results':
        return this.generateLabResultsResponse(request, caseInfo, config);
      
      case 'imaging':
        return this.generateImagingResponse(request, caseInfo, config);
      
      case 'specialist_consultation':
        return this.generateSpecialistResponse(request, caseInfo, config);
      
      default:
        return {
          granted: false,
          information: "Unknown request type.",
          cost: 0
        };
    }
  }

  private generateHistoryResponse(
    request: InformationRequest,
    caseInfo: CaseInformation
  ): InformationResponse {
    let information = "";
    let cost = 0;

    if (request.specificRequest.toLowerCase().includes('present illness')) {
      information = caseInfo.historyOfPresentIllness || "No additional history of present illness available.";
      cost = 0; // History taking is typically free
    } else if (request.specificRequest.toLowerCase().includes('past medical')) {
      information = caseInfo.pastMedicalHistory?.join(', ') || "No significant past medical history.";
      cost = 0;
    } else if (request.specificRequest.toLowerCase().includes('medication')) {
      information = caseInfo.medications?.join(', ') || "No current medications.";
      cost = 0;
    } else if (request.specificRequest.toLowerCase().includes('family')) {
      information = caseInfo.familyHistory?.join(', ') || "No significant family history.";
      cost = 0;
    } else if (request.specificRequest.toLowerCase().includes('social')) {
      information = caseInfo.socialHistory || "No significant social history.";
      cost = 0;
    } else {
      information = "Please be more specific about which aspect of the history you need.";
      cost = 0;
    }

    return {
      granted: true,
      information,
      cost,
      followUpQuestions: this.generateFollowUpQuestions(request.requestType)
    };
  }

  private async generatePhysicalExamResponse(
    request: InformationRequest,
    caseInfo: CaseInformation,
    _config: RunnableConfig
  ): Promise<InformationResponse> {
    // Generate realistic physical exam findings
    const model = await this.loadChatModel("gpt-4");
    
    const examPrompt = `
Generate realistic physical examination findings for a patient with:
- Chief complaint: ${caseInfo.chiefComplaint}
- Age: ${caseInfo.demographics.age}
- Gender: ${caseInfo.demographics.gender}

Specific exam requested: ${request.specificRequest}

Provide realistic findings that could be consistent with the chief complaint.
Be specific and clinically relevant.
`;

    const response = await model.invoke([new HumanMessage(examPrompt)]);
    
    return {
      granted: true,
      information: response.content,
      cost: 0, // Physical exam is typically part of the visit
      additionalContext: "Physical examination performed during current visit."
    };
  }

  private async generateLabResultsResponse(
    request: InformationRequest,
    caseInfo: CaseInformation,
    _config: RunnableConfig
  ): Promise<InformationResponse> {
    // Generate synthetic lab results
    const model = await this.loadChatModel("gpt-4");
    
    const labPrompt = `
Generate realistic laboratory results for:
- Chief complaint: ${caseInfo.chiefComplaint}
- Age: ${caseInfo.demographics.age}
- Gender: ${caseInfo.demographics.gender}

Specific lab test requested: ${request.specificRequest}

Provide realistic values with normal ranges.
Make results clinically relevant to the presentation.
`;

    const response = await model.invoke([new HumanMessage(labPrompt)]);
    
    const cost = this.calculateLabCost(request.specificRequest);
    
    return {
      granted: true,
      information: response.content,
      cost,
      additionalContext: "Laboratory results available within 2-4 hours."
    };
  }

  private async generateImagingResponse(
    request: InformationRequest,
    caseInfo: CaseInformation,
    _config: RunnableConfig
  ): Promise<InformationResponse> {
    // Generate synthetic imaging results
    const model = await this.loadChatModel("gpt-4");
    
    const imagingPrompt = `
Generate a realistic imaging report for:
- Chief complaint: ${caseInfo.chiefComplaint}
- Age: ${caseInfo.demographics.age}
- Gender: ${caseInfo.demographics.gender}

Imaging study requested: ${request.specificRequest}

Provide a structured radiology report with:
- Technique
- Findings
- Impression
Make findings clinically relevant and realistic.
`;

    const response = await model.invoke([new HumanMessage(imagingPrompt)]);
    
    const cost = this.calculateImagingCost(request.specificRequest);
    
    return {
      granted: true,
      information: response.content,
      cost,
      additionalContext: "Imaging results available within 4-24 hours depending on urgency."
    };
  }

  private async generateSpecialistResponse(
    request: InformationRequest,
    caseInfo: CaseInformation,
    _config: RunnableConfig
  ): Promise<InformationResponse> {
    // Generate specialist consultation note
    const model = await this.loadChatModel("gpt-4");
    
    const consultPrompt = `
Generate a specialist consultation note for:
- Chief complaint: ${caseInfo.chiefComplaint}
- Age: ${caseInfo.demographics.age}
- Gender: ${caseInfo.demographics.gender}

Specialist consultation requested: ${request.specificRequest}

Provide a structured consultation note with:
- Assessment
- Recommendations
- Follow-up plan
Make recommendations clinically appropriate.
`;

    const response = await model.invoke([new HumanMessage(consultPrompt)]);
    
    return {
      granted: true,
      information: response.content,
      cost: 300, // Typical specialist consultation cost
      additionalContext: "Specialist consultation typically available within 1-2 weeks."
    };
  }

  private calculateLabCost(testName: string): number {
    const testCosts: Record<string, number> = {
      'cbc': 25,
      'complete blood count': 25,
      'bmp': 30,
      'basic metabolic panel': 30,
      'cmp': 40,
      'comprehensive metabolic panel': 40,
      'lipid panel': 35,
      'liver function': 45,
      'thyroid function': 60,
      'cardiac enzymes': 80,
      'troponin': 50,
      'pt/inr': 25,
      'ptt': 25,
      'urinalysis': 20,
      'blood culture': 75,
      'urine culture': 50
    };

    const testLower = testName.toLowerCase();
    for (const [test, cost] of Object.entries(testCosts)) {
      if (testLower.includes(test)) {
        return cost;
      }
    }
    
    return 50; // Default lab cost
  }

  private calculateImagingCost(imagingType: string): number {
    const imagingCosts: Record<string, number> = {
      'chest x-ray': 150,
      'abdominal x-ray': 200,
      'ct head': 800,
      'ct chest': 900,
      'ct abdomen': 1000,
      'ct pelvis': 1000,
      'mri brain': 1500,
      'mri spine': 1600,
      'ultrasound': 300,
      'echocardiogram': 500,
      'ekg': 50,
      'ecg': 50
    };

    const imagingLower = imagingType.toLowerCase();
    for (const [imaging, cost] of Object.entries(imagingCosts)) {
      if (imagingLower.includes(imaging)) {
        return cost;
      }
    }
    
    return 400; // Default imaging cost
  }

  private generateFollowUpQuestions(requestType: string): string[] {
    const questions: Record<string, string[]> = {
      'history': [
        "Any recent changes in symptoms?",
        "Any family history of similar conditions?",
        "Any recent travel or exposures?"
      ],
      'physical_exam': [
        "Any areas of tenderness not yet examined?",
        "Any functional limitations?",
        "Any changes since last examination?"
      ],
      'lab_results': [
        "Should we trend these values?",
        "Any additional tests needed based on results?",
        "How do these compare to previous values?"
      ],
      'imaging': [
        "Any additional views needed?",
        "Should we consider contrast studies?",
        "Are comparison studies available?"
      ]
    };

    return questions[requestType] || [];
  }

  public getRevealedInformation(): string[] {
    return Array.from(this.revealedSections);
  }

  /**
   * Process user responses to patient questions and update case information
   */
  async processUserResponse(
    state: MedicalDiagnosticStateType,
    userResponse: string,
    config: RunnableConfig
  ): Promise<CaseInformation> {
    const configuration = ensureMedicalConfiguration(config);
    const model = await this.loadChatModel(configuration.model);

    // Extract structured information from user response
    const extractionPrompt = `
You are a medical information processor. Extract and structure information from this patient response:

Patient Response: "${userResponse}"

Original Case Information:
- Chief Complaint: ${state.availableCaseInfo.chiefComplaint}
- Age: ${state.availableCaseInfo.demographics.age}
- Gender: ${state.availableCaseInfo.demographics.gender}
- Current History: ${state.availableCaseInfo.historyOfPresentIllness || 'Limited'}

Pending Questions that might have been answered:
${state.pendingQuestions.map((q, i) => `${i + 1}. ${q.question} (Category: ${q.category})`).join('\n')}

Extract and categorize new information from the patient's response. Update the case information accordingly.

Return updated case information in JSON format:
{
  "historyOfPresentIllness": "enhanced history including new details",
  "pastMedicalHistory": ["condition1", "condition2"],
  "medications": ["med1", "med2"],
  "allergies": ["allergy1"],
  "familyHistory": ["family condition1"],
  "socialHistory": "social details",
  "reviewOfSystems": {
    "cardiovascular": "details",
    "respiratory": "details",
    "gastrointestinal": "details"
  },
  "physicalExam": {
    "general": "details",
    "vitals": "details"
  }
}

Only include fields that have new information. Preserve existing information and add new details.
`;

    const response = await model.invoke([new HumanMessage(extractionPrompt)]);
    
    try {
      const extractedInfo = JSON.parse(response.content as string);
      
      // Merge with existing case information
      const updatedCaseInfo: CaseInformation = {
        ...state.availableCaseInfo,
        ...extractedInfo,
        pastMedicalHistory: [
          ...(state.availableCaseInfo.pastMedicalHistory || []),
          ...(extractedInfo.pastMedicalHistory || [])
        ].filter((item, index, array) => array.indexOf(item) === index), // Remove duplicates
        medications: [
          ...(state.availableCaseInfo.medications || []),
          ...(extractedInfo.medications || [])
        ].filter((item, index, array) => array.indexOf(item) === index),
        allergies: [
          ...(state.availableCaseInfo.allergies || []),
          ...(extractedInfo.allergies || [])
        ].filter((item, index, array) => array.indexOf(item) === index),
        familyHistory: [
          ...(state.availableCaseInfo.familyHistory || []),
          ...(extractedInfo.familyHistory || [])
        ].filter((item, index, array) => array.indexOf(item) === index),
        reviewOfSystems: {
          ...state.availableCaseInfo.reviewOfSystems,
          ...extractedInfo.reviewOfSystems
        },
        physicalExam: {
          ...state.availableCaseInfo.physicalExam,
          ...extractedInfo.physicalExam
        }
      };

      return updatedCaseInfo;
    } catch (error) {
      // Fallback: append user response to history of present illness
      const enhancedHistory = state.availableCaseInfo.historyOfPresentIllness 
        ? `${state.availableCaseInfo.historyOfPresentIllness}\n\nAdditional Information: ${userResponse}`
        : `Additional Information: ${userResponse}`;

      return {
        ...state.availableCaseInfo,
        historyOfPresentIllness: enhancedHistory
      };
    }
  }

  /**
   * Validate user response completeness
   */
  validateUserResponse(
    userResponse: string,
    pendingQuestions: string[]
  ): { isComplete: boolean; missingQuestions: string[] } {
    // Simple validation - check if response addresses the questions
    const responseWords = userResponse.toLowerCase().split(/\s+/);
    const missingQuestions: string[] = [];

    pendingQuestions.forEach((question, index) => {
      const questionKeywords = this.extractQuestionKeywords(question);
      const hasResponse = questionKeywords.some(keyword => 
        responseWords.some(word => word.includes(keyword.toLowerCase()))
      );
      
      if (!hasResponse) {
        missingQuestions.push(`Question ${index + 1}: ${question}`);
      }
    });

    return {
      isComplete: missingQuestions.length === 0,
      missingQuestions
    };
  }

  private extractQuestionKeywords(question: string): string[] {
    // Extract key medical terms from questions
    const medicalKeywords = [
      'pain', 'fever', 'nausea', 'vomiting', 'headache', 'fatigue',
      'medications', 'allergies', 'surgery', 'hospital', 'family',
      'smoking', 'drinking', 'travel', 'weight', 'appetite',
      'sleep', 'bowel', 'urination', 'breathing', 'chest'
    ];

    const questionWords = question.toLowerCase().split(/\s+/);
    return questionWords.filter(word => 
      medicalKeywords.some(keyword => word.includes(keyword)) ||
      word.length > 4 // Include longer words that might be medical terms
    );
  }

  public resetGatekeeper(): void {
    this.revealedSections.clear();
  }
}