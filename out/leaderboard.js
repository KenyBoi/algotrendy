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
exports.LeaderboardProvider = void 0;
const vscode = __importStar(require("vscode"));
class LeaderboardProvider {
    constructor(context) {
        this.LEADERBOARD_KEY = 'promptOrDie.leaderboard';
        this.USER_SCORE_KEY = 'promptOrDie.userScore';
        this.context = context;
    }
    async addScore(points) {
        const currentScore = await this.getCurrentUserScore();
        const newScore = currentScore + points;
        await this.context.globalState.update(this.USER_SCORE_KEY, newScore);
        // Update leaderboard
        const leaderboard = await this.getLeaderboard();
        const username = await this.getUsername();
        const existingEntry = leaderboard.find(entry => entry.username === username);
        if (existingEntry) {
            existingEntry.totalScore = newScore;
            existingEntry.challengesCompleted += 1;
            existingEntry.lastPlayed = new Date();
            existingEntry.averageScore = existingEntry.totalScore / existingEntry.challengesCompleted;
        }
        else {
            leaderboard.push({
                username,
                totalScore: newScore,
                challengesCompleted: 1,
                lastPlayed: new Date(),
                averageScore: newScore
            });
        }
        // Sort by total score (descending)
        leaderboard.sort((a, b) => b.totalScore - a.totalScore);
        // Keep only top 50 entries
        const topEntries = leaderboard.slice(0, 50);
        await this.context.globalState.update(this.LEADERBOARD_KEY, topEntries);
    }
    async getCurrentUserScore() {
        return this.context.globalState.get(this.USER_SCORE_KEY, 0);
    }
    async getTopScores(limit = 10) {
        const leaderboard = await this.getLeaderboard();
        return leaderboard.slice(0, limit);
    }
    async getLeaderboard() {
        return this.context.globalState.get(this.LEADERBOARD_KEY, []);
    }
    async getUsername() {
        // Try to get username from git config
        try {
            const gitExtension = vscode.extensions.getExtension('vscode.git');
            if (gitExtension && gitExtension.isActive) {
                const git = gitExtension.exports.getAPI(1);
                const repositories = git.repositories;
                if (repositories.length > 0) {
                    const config = repositories[0].state.HEAD?.name || 'Unknown';
                    return config;
                }
            }
        }
        catch (error) {
            // Fallback to asking user
        }
        // Ask user for username if not available
        const username = await vscode.window.showInputBox({
            prompt: 'Enter your username for the leaderboard',
            placeHolder: 'Your username',
            value: 'Anonymous'
        });
        return username || 'Anonymous';
    }
    async resetUserScore() {
        await this.context.globalState.update(this.USER_SCORE_KEY, 0);
        vscode.window.showInformationMessage('Your score has been reset to 0.');
    }
    async exportLeaderboard() {
        const leaderboard = await this.getLeaderboard();
        const csvContent = this.convertToCSV(leaderboard);
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('prompt-or-die-leaderboard.csv'),
            filters: {
                'CSV Files': ['csv'],
                'All Files': ['*']
            }
        });
        if (uri) {
            await vscode.workspace.fs.writeFile(uri, Buffer.from(csvContent, 'utf8'));
            vscode.window.showInformationMessage(`Leaderboard exported to ${uri.fsPath}`);
        }
    }
    convertToCSV(leaderboard) {
        const headers = ['Rank', 'Username', 'Total Score', 'Challenges Completed', 'Average Score', 'Last Played'];
        const rows = leaderboard.map((entry, index) => [
            (index + 1).toString(),
            entry.username || 'Anonymous',
            entry.totalScore.toString(),
            entry.challengesCompleted.toString(),
            entry.averageScore.toFixed(2),
            entry.lastPlayed.toISOString().split('T')[0]
        ]);
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
    async getUserRank() {
        const leaderboard = await this.getLeaderboard();
        const username = await this.getUsername();
        const userEntry = leaderboard.find(entry => entry.username === username);
        if (!userEntry) {
            return -1; // User not in leaderboard
        }
        return leaderboard.indexOf(userEntry) + 1;
    }
    async getLeaderboardStats() {
        const leaderboard = await this.getLeaderboard();
        if (leaderboard.length === 0) {
            return { totalPlayers: 0, averageScore: 0, topScore: 0 };
        }
        const totalPlayers = leaderboard.length;
        const averageScore = leaderboard.reduce((sum, entry) => sum + entry.totalScore, 0) / totalPlayers;
        const topScore = leaderboard[0]?.totalScore || 0;
        return { totalPlayers, averageScore, topScore };
    }
}
exports.LeaderboardProvider = LeaderboardProvider;
//# sourceMappingURL=leaderboard.js.map