import 'dotenv/config';

function required(name: string, fallback?: string): string {
  const v = process.env[name] ?? fallback;
  if (v === undefined) throw new Error(`missing env ${name}`);
  return v;
}

export const env = {
  port: Number(process.env.PORT ?? 8081),
  aiServiceUrl: required('AI_SERVICE_URL', 'http://localhost:8080'),
  postgresDsn: required('POSTGRES_DSN', 'postgresql://expenseiq:expenseiq@localhost:5432/expenseiq'),
  mongoUri: required('MONGO_URI', 'mongodb://localhost:27017'),
  mongoDb: required('MONGO_DB', 'expenseiq'),
  logLevel: process.env.LOG_LEVEL ?? 'info',
};
