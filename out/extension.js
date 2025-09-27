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
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const challengeManager_1 = require("./challengeManager");
const leaderboard_1 = require("./leaderboard");
let challengeManager;
let leaderboardProvider;
function activate(context) {
    console.log('Prompt or Die extension is now active!');
    // Initialize managers
    challengeManager = new challengeManager_1.ChallengeManager(context);
    leaderboardProvider = new leaderboard_1.LeaderboardProvider(context);
    // Register commands
    const startChallengeCommand = vscode.commands.registerCommand('promptOrDie.startChallenge', async () => {
        await startNewChallenge();
    });
    const showLeaderboardCommand = vscode.commands.registerCommand('promptOrDie.showLeaderboard', async () => {
        await showLeaderboard();
    });
    const settingsCommand = vscode.commands.registerCommand('promptOrDie.settings', async () => {
        await vscode.commands.executeCommand('workbench.action.openSettings', 'promptOrDie');
    });
    // Add to subscriptions
    context.subscriptions.push(startChallengeCommand, showLeaderboardCommand, settingsCommand);
    // Show welcome message
    vscode.window.showInformationMessage('Welcome to Prompt or Die! Ready to test your AI prompting skills?', 'Start Challenge', 'Learn More')
        .then(selection => {
        if (selection === 'Start Challenge') {
            vscode.commands.executeCommand('promptOrDie.startChallenge');
        }
        else if (selection === 'Learn More') {
            vscode.env.openExternal(vscode.Uri.parse('https://github.com/KenyBoi/algotrendy'));
        }
    });
}
exports.activate = activate;
async function startNewChallenge() {
    try {
        const difficulty = vscode.workspace.getConfiguration('promptOrDie').get('difficulty', 'normal');
        const challenge = await challengeManager.getRandomChallenge(difficulty);
        if (challenge) {
            await showChallengePanel(challenge);
        }
        else {
            vscode.window.showErrorMessage('No challenges available for the selected difficulty level.');
        }
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to start challenge: ${error}`);
    }
}
async function showChallengePanel(challenge) {
    const panel = vscode.window.createWebviewPanel('promptOrDieChallenge', 'Prompt or Die Challenge', vscode.ViewColumn.One, {
        enableScripts: true,
        retainContextWhenHidden: true
    });
    panel.webview.html = getChallengeWebviewContent(challenge);
    // Handle messages from webview
    panel.webview.onDidReceiveMessage(async (message) => {
        switch (message.command) {
            case 'submitAnswer':
                const result = await challengeManager.evaluateAnswer(challenge.id, message.answer);
                panel.webview.postMessage({ command: 'showResult', result });
                if (result.correct) {
                    await leaderboardProvider.addScore(result.score);
                    vscode.window.showInformationMessage(`Correct! You earned ${result.score} points!`);
                }
                else {
                    vscode.window.showWarningMessage('Incorrect answer. Try again!');
                }
                break;
            case 'getHint':
                const hint = await challengeManager.getHint(challenge.id);
                panel.webview.postMessage({ command: 'showHint', hint });
                break;
        }
    });
}
async function showLeaderboard() {
    const scores = await leaderboardProvider.getTopScores();
    const panel = vscode.window.createWebviewPanel('promptOrDieLeaderboard', 'Prompt or Die Leaderboard', vscode.ViewColumn.One, {
        enableScripts: true
    });
    panel.webview.html = getLeaderboardWebviewContent(scores);
}
function getChallengeWebviewContent(challenge) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt or Die Challenge</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
            line-height: 1.6;
        }
        .challenge-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .challenge-title {
            color: var(--vscode-textLink-foreground);
            border-bottom: 2px solid var(--vscode-textLink-foreground);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .challenge-description {
            background-color: var(--vscode-textBlockQuote-background);
            border-left: 4px solid var(--vscode-textBlockQuote-border);
            padding: 15px;
            margin: 20px 0;
        }
        .prompt-input {
            width: 100%;
            height: 150px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 10px;
            font-family: var(--vscode-editor-font-family);
            font-size: var(--vscode-editor-font-size);
            resize: vertical;
        }
        .button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 10px 20px;
            margin: 10px 5px;
            cursor: pointer;
            border-radius: 3px;
        }
        .button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }
        .hint-button {
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        .difficulty-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .difficulty-${challenge.difficulty} {
            background-color: ${getDifficultyColor(challenge.difficulty)};
            color: white;
        }
        .result-panel {
            display: none;
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
        }
        .result-correct {
            background-color: var(--vscode-terminal-ansiGreen);
            color: var(--vscode-terminal-background);
        }
        .result-incorrect {
            background-color: var(--vscode-terminal-ansiRed);
            color: var(--vscode-terminal-background);
        }
    </style>
</head>
<body>
    <div class="challenge-container">
        <h1 class="challenge-title">${challenge.title}</h1>
        <div class="difficulty-badge difficulty-${challenge.difficulty}">
            ${challenge.difficulty.toUpperCase()}
        </div>
        
        <div class="challenge-description">
            <h3>Challenge:</h3>
            <p>${challenge.description}</p>
            
            <h3>Goal:</h3>
            <p>${challenge.goal}</p>
            
            ${challenge.context ? `<h3>Context:</h3><p>${challenge.context}</p>` : ''}
        </div>

        <div>
            <h3>Your Prompt:</h3>
            <textarea id="promptInput" class="prompt-input" placeholder="Enter your AI prompt here..."></textarea>
        </div>

        <div>
            <button class="button" onclick="submitAnswer()">Submit Answer</button>
            <button class="button hint-button" onclick="getHint()">Get Hint</button>
        </div>

        <div id="resultPanel" class="result-panel">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function submitAnswer() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) {
                alert('Please enter a prompt before submitting!');
                return;
            }
            vscode.postMessage({ command: 'submitAnswer', answer: prompt });
        }

        function getHint() {
            vscode.postMessage({ command: 'getHint' });
        }

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.command) {
                case 'showResult':
                    showResult(message.result);
                    break;
                case 'showHint':
                    alert('Hint: ' + message.hint);
                    break;
            }
        });

        function showResult(result) {
            const panel = document.getElementById('resultPanel');
            const content = document.getElementById('resultContent');
            
            panel.className = 'result-panel ' + (result.correct ? 'result-correct' : 'result-incorrect');
            content.innerHTML = \`
                <h3>\${result.correct ? '✅ Correct!' : '❌ Incorrect'}</h3>
                <p><strong>Score:</strong> \${result.score} points</p>
                <p><strong>Feedback:</strong> \${result.feedback}</p>
                \${result.explanation ? \`<p><strong>Explanation:</strong> \${result.explanation}</p>\` : ''}
            \`;
            
            panel.style.display = 'block';
        }
    </script>
</body>
</html>`;
}
function getLeaderboardWebviewContent(scores) {
    const scoreRows = scores.map((score, index) => `<tr>
            <td>${index + 1}</td>
            <td>${score.username || 'Anonymous'}</td>
            <td>${score.totalScore}</td>
            <td>${score.challengesCompleted}</td>
            <td>${new Date(score.lastPlayed).toLocaleDateString()}</td>
        </tr>`).join('');
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaderboard</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        th {
            background-color: var(--vscode-editor-lineHighlightBackground);
            font-weight: bold;
        }
        .rank-1 { color: #FFD700; }
        .rank-2 { color: #C0C0C0; }
        .rank-3 { color: #CD7F32; }
    </style>
</head>
<body>
    <h1>🏆 Prompt or Die Leaderboard</h1>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Score</th>
                <th>Challenges</th>
                <th>Last Played</th>
            </tr>
        </thead>
        <tbody>
            ${scoreRows}
        </tbody>
    </table>
</body>
</html>`;
}
function getDifficultyColor(difficulty) {
    switch (difficulty) {
        case 'easy': return '#4CAF50';
        case 'normal': return '#FF9800';
        case 'hard': return '#F44336';
        case 'expert': return '#9C27B0';
        default: return '#757575';
    }
}
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map