"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProgressTracker = void 0;
const vscode = __importStar(require("vscode"));
class ProgressTracker {
    constructor(context) {
        this.PROGRESS_KEY = 'promptOrDie.userProgress';
        this.context = context;
    }
    async getUserProgress() {
        const defaultProgress = {
            userId: await this.getUserId(),
            completedChallenges: [],
            totalScore: 0,
            currentStreak: 0,
            longestStreak: 0,
            achievements: []
        };
        return this.context.globalState.get(this.PROGRESS_KEY, defaultProgress);
    }
    async updateProgress(challengeId, score) {
        const progress = await this.getUserProgress();
        if (!progress.completedChallenges.includes(challengeId)) {
            progress.completedChallenges.push(challengeId);
            progress.currentStreak += 1;
            progress.longestStreak = Math.max(progress.longestStreak, progress.currentStreak);
        }
        progress.totalScore += score;
        // Check for new achievements
        await this.checkAchievements(progress);
        await this.context.globalState.update(this.PROGRESS_KEY, progress);
    }
    async checkAchievements(progress) {
        const newAchievements = [];
        // First Steps achievement
        if (progress.completedChallenges.length === 1 && !this.hasAchievement(progress, 'first-steps')) {
            newAchievements.push({
                id: 'first-steps',
                name: 'First Steps',
                description: 'Complete your first challenge',
                icon: '🎯',
                unlockedAt: new Date()
            });
        }
        // Marathon Runner achievement
        if (progress.completedChallenges.length >= 50 && !this.hasAchievement(progress, 'marathon-runner')) {
            newAchievements.push({
                id: 'marathon-runner',
                name: 'Marathon Runner',
                description: 'Complete 50 challenges',
                icon: '🏃‍♂️',
                unlockedAt: new Date()
            });
        }
        // Streak Master achievement
        if (progress.longestStreak >= 10 && !this.hasAchievement(progress, 'streak-master')) {
            newAchievements.push({
                id: 'streak-master',
                name: 'Streak Master',
                description: 'Achieve a 10-challenge streak',
                icon: '🔥',
                unlockedAt: new Date()
            });
        }
        // High Scorer achievement
        if (progress.totalScore >= 5000 && !this.hasAchievement(progress, 'high-scorer')) {
            newAchievements.push({
                id: 'high-scorer',
                name: 'High Scorer',
                description: 'Reach 5000 total points',
                icon: '⭐',
                unlockedAt: new Date()
            });
        }
        // Add new achievements to progress
        progress.achievements.push(...newAchievements);
        // Show achievement notifications
        for (const achievement of newAchievements) {
            vscode.window.showInformationMessage(`🎉 Achievement Unlocked: ${achievement.name}!`, 'View Achievements').then(selection => {
                if (selection === 'View Achievements') {
                    this.showAchievements();
                }
            });
        }
    }
    hasAchievement(progress, achievementId) {
        return progress.achievements.some(a => a.id === achievementId);
    }
    async getUserId() {
        // Generate a simple user ID based on workspace or use a random one
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            return workspaceFolders[0].uri.fsPath.replace(/[^a-zA-Z0-9]/g, '');
        }
        return Math.random().toString(36).substring(2, 15);
    }
    async showAchievements() {
        const progress = await this.getUserProgress();
        const panel = vscode.window.createWebviewPanel('promptOrDieAchievements', 'Achievements', vscode.ViewColumn.One, { enableScripts: true });
        panel.webview.html = this.getAchievementsWebviewContent(progress.achievements);
    }
    getAchievementsWebviewContent(achievements) {
        const achievementsList = achievements.map(achievement => `
            <div class="achievement">
                <div class="achievement-icon">${achievement.icon}</div>
                <div class="achievement-info">
                    <h3>${achievement.name}</h3>
                    <p>${achievement.description}</p>
                    <small>Unlocked: ${achievement.unlockedAt.toLocaleDateString()}</small>
                </div>
            </div>
        `).join('');
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Achievements</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }
        .achievements-container {
            max-width: 600px;
            margin: 0 auto;
        }
        .achievement {
            display: flex;
            align-items: center;
            background-color: var(--vscode-editor-lineHighlightBackground);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .achievement-icon {
            font-size: 48px;
            margin-right: 20px;
        }
        .achievement-info h3 {
            margin: 0 0 5px 0;
            color: var(--vscode-textLink-foreground);
        }
        .achievement-info p {
            margin: 0 0 5px 0;
            opacity: 0.8;
        }
        .achievement-info small {
            opacity: 0.6;
        }
        .no-achievements {
            text-align: center;
            color: var(--vscode-descriptionForeground);
            font-style: italic;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="achievements-container">
        <h1>🏆 Your Achievements</h1>
        ${achievements.length > 0 ? achievementsList : '<div class="no-achievements">No achievements yet. Start completing challenges to unlock them!</div>'}
    </div>
</body>
</html>`;
    }
    async resetProgress() {
        const result = await vscode.window.showWarningMessage('Are you sure you want to reset all progress? This cannot be undone.', 'Reset', 'Cancel');
        if (result === 'Reset') {
            await this.context.globalState.update(this.PROGRESS_KEY, undefined);
            vscode.window.showInformationMessage('Progress has been reset.');
        }
    }
}
exports.ProgressTracker = ProgressTracker;
//# sourceMappingURL=progressTracker.js.map