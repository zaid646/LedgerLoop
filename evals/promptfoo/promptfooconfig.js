const baseUrl = process.env.VLLM_BASE_URL;
const model = process.env.VLLM_MODEL;

module.exports = {
  description: 'ExpenseIQ — receipt extraction regression suite (PromptFoo)',
  prompts: ['file://prompt.json'],
  providers: [
    {
      id: `openai:chat:${model}`,
      label: 'vllm-receipt-extraction',
      config: {
        apiBaseUrl: baseUrl,
        apiKey: process.env.VLLM_API_KEY || 'dummy',
        temperature: 0,
        max_tokens: 700,
      },
    },
  ],
  defaultTest: {
    assert: [
      { type: 'is-json' },
      { type: 'javascript', value: 'file://assert_receipt.js' },
    ],
  },
  tests: 'file://tests.jsonl',
};
