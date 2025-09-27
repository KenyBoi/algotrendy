"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChallengeManager = void 0;
class ChallengeManager {
    constructor(context) {
        this.challenges = [];
        this.context = context;
        this.loadChallenges();
    }
    loadChallenges() {
        // Sample challenges - in a real implementation, these might come from a file or API
        this.challenges = [
            {
                id: 'prompt-basics-1',
                title: 'Basic Task Instruction',
                description: 'Create a prompt that instructs an AI to write a professional email',
                goal: 'Write a prompt that results in a well-structured, professional email with proper greeting, body, and closing',
                difficulty: 'easy',
                category: 'basics',
                points: 100,
                hints: [
                    'Be specific about the email\'s purpose',
                    'Include context about who the email is for',
                    'Specify the tone and format'
                ],
                expectedKeywords: ['email', 'professional', 'formal'],
                rubric: {
                    criteria: ['Clarity', 'Specificity', 'Context'],
                    maxScore: 100,
                    passingScore: 70
                }
            },
            {
                id: 'context-setting-1',
                title: 'Context is King',
                description: 'Create a prompt that provides sufficient context for a complex task',
                goal: 'Write a prompt that helps an AI understand a multi-step process with proper context',
                context: 'You need to explain how to debug a JavaScript application to a new developer',
                difficulty: 'normal',
                category: 'context',
                points: 150,
                hints: [
                    'Include background information',
                    'Define technical terms',
                    'Break down complex steps'
                ],
                expectedKeywords: ['debug', 'javascript', 'step-by-step'],
                rubric: {
                    criteria: ['Context Clarity', 'Step Organization', 'Technical Accuracy'],
                    maxScore: 150,
                    passingScore: 105
                }
            },
            {
                id: 'role-playing-1',
                title: 'AI Persona Challenge',
                description: 'Create a prompt that establishes a specific AI persona/role',
                goal: 'Design a prompt that makes the AI adopt a specific professional role with appropriate expertise and communication style',
                difficulty: 'hard',
                category: 'role-playing',
                points: 200,
                hints: [
                    'Define the role clearly',
                    'Specify expertise areas',
                    'Include communication style preferences'
                ],
                expectedKeywords: ['role', 'expert', 'persona'],
                rubric: {
                    criteria: ['Role Definition', 'Expertise Scope', 'Style Consistency'],
                    maxScore: 200,
                    passingScore: 140
                }
            },
            {
                id: 'chain-of-thought-1',
                title: 'Reasoning Master',
                description: 'Create a prompt that encourages step-by-step reasoning',
                goal: 'Design a prompt that makes the AI show its reasoning process for a complex problem',
                context: 'The AI needs to solve a logical puzzle or mathematical problem',
                difficulty: 'expert',
                category: 'reasoning',
                points: 300,
                hints: [
                    'Request explicit reasoning steps',
                    'Ask for assumptions to be stated',
                    'Encourage verification of each step'
                ],
                expectedKeywords: ['step-by-step', 'reasoning', 'explain'],
                rubric: {
                    criteria: ['Reasoning Clarity', 'Step Structure', 'Problem Decomposition'],
                    maxScore: 300,
                    passingScore: 210
                }
            }
        ];
    }
    async getRandomChallenge(difficulty) {
        const filteredChallenges = this.challenges.filter(c => c.difficulty === difficulty);
        if (filteredChallenges.length === 0) {
            return null;
        }
        const randomIndex = Math.floor(Math.random() * filteredChallenges.length);
        return filteredChallenges[randomIndex];
    }
    async getChallengeById(id) {
        return this.challenges.find(c => c.id === id) || null;
    }
    async evaluateAnswer(challengeId, userPrompt) {
        const challenge = await this.getChallengeById(challengeId);
        if (!challenge) {
            return {
                correct: false,
                score: 0,
                feedback: 'Challenge not found'
            };
        }
        // Simple evaluation logic - in a real implementation, this might use AI to evaluate
        const score = this.calculateScore(challenge, userPrompt);
        const correct = score >= challenge.rubric.passingScore;
        return {
            correct,
            score,
            feedback: this.generateFeedback(challenge, userPrompt, score),
            explanation: correct ? 'Great job! Your prompt meets the requirements.' : 'Your prompt needs improvement in some areas.'
        };
    }
    calculateScore(challenge, userPrompt) {
        let score = 0;
        const prompt = userPrompt.toLowerCase();
        // Check for expected keywords (basic scoring)
        if (challenge.expectedKeywords) {
            const keywordMatches = challenge.expectedKeywords.filter(keyword => prompt.includes(keyword.toLowerCase())).length;
            score += (keywordMatches / challenge.expectedKeywords.length) * 50;
        }
        // Check prompt length (reasonable length gets points)
        if (userPrompt.length >= 50 && userPrompt.length <= 500) {
            score += 20;
        }
        // Check for question marks or clear instructions
        if (prompt.includes('?') || prompt.includes('please') || prompt.includes('help')) {
            score += 15;
        }
        // Check for context setting
        if (prompt.includes('context') || prompt.includes('background') || prompt.includes('situation')) {
            score += 15;
        }
        return Math.min(score, challenge.rubric.maxScore);
    }
    generateFeedback(challenge, userPrompt, score) {
        const feedbacks = [];
        if (score < 50) {
            feedbacks.push('Your prompt needs more specific instructions and context.');
        }
        else if (score < 100) {
            feedbacks.push('Good start! Consider adding more detail and context.');
        }
        else if (score < 150) {
            feedbacks.push('Well done! Your prompt is clear and specific.');
        }
        else {
            feedbacks.push('Excellent! Your prompt demonstrates mastery of the concept.');
        }
        // Add specific suggestions based on challenge
        if (challenge.expectedKeywords) {
            const missingKeywords = challenge.expectedKeywords.filter(keyword => !userPrompt.toLowerCase().includes(keyword.toLowerCase()));
            if (missingKeywords.length > 0) {
                feedbacks.push(`Consider including: ${missingKeywords.join(', ')}`);
            }
        }
        return feedbacks.join(' ');
    }
    async getHint(challengeId) {
        const challenge = await this.getChallengeById(challengeId);
        if (!challenge || challenge.hints.length === 0) {
            return 'No hints available for this challenge.';
        }
        // Return a random hint
        const randomIndex = Math.floor(Math.random() * challenge.hints.length);
        return challenge.hints[randomIndex];
    }
    getChallengesByDifficulty(difficulty) {
        return this.challenges.filter(c => c.difficulty === difficulty);
    }
    getAllChallenges() {
        return [...this.challenges];
    }
}
exports.ChallengeManager = ChallengeManager;
//# sourceMappingURL=challengeManager.js.map