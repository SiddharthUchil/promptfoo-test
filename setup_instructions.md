# Setup Instructions

## 🔑 **Required: Set OpenAI API Key**

The evaluation failed because the OpenAI API key is not properly set. Here's how to fix it:

### Option 1: Create .env file (Recommended)

```bash
# Create .env file in the promptfoo_architecture folder
echo "OPENAI_API_KEY=sk-your-actual-openai-api-key-here" > .env
```

### Option 2: Export environment variable

```bash
export OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### Option 3: Set in your shell profile

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

## 🧪 **Test Setup**

```bash
python test_setup.py
```

## 🚀 **Run Evaluation**

```bash
npm run eval
```

## 📊 **What Happened in Your Last Run**

The evaluation **partially worked** but failed due to API authentication:

- ✅ Promptfoo CLI ran successfully
- ✅ Configuration was loaded correctly
- ✅ Test cases were processed
- ❌ All 10 API calls failed with "401 Unauthorized"
- ❌ Results show 0% pass rate due to missing API key

## 🔧 **Current Status**

Your setup is almost complete! You just need to:

1. Set your OpenAI API key (instructions above)
2. Re-run the evaluation

The system is working correctly - the error messages show that promptfoo processed your configuration and test cases properly, but couldn't access the OpenAI API without valid credentials.

## 🎯 **Expected Results After Fixing API Key**

Once you set the API key, you'll get:

- Excel file with all results: `evaluation_results_latest.xlsx`
- RAGAS metrics for each response
- F1 scores for category classification
- Comparison between Prompt V1 and V2
- Professional summary report

## 🛠️ **Troubleshooting**

If you still have issues after setting the API key:

1. Verify your API key is valid: https://platform.openai.com/api-keys
2. Check your OpenAI account has available credits
3. Restart your terminal/shell after setting environment variables
4. Run `echo $OPENAI_API_KEY` to verify it's set correctly
