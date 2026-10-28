import { handler } from "./main.ts";

const syncEnv = (env: Record<string, string>) => {
  Object.entries(env).forEach(([k, v]) => {
    if (typeof v === "string") {
      Deno.env.set(k, v);
    }
  });
};

export const worker = {
  fetch: (req: Request, env: Record<string, string>) => {
    syncEnv(env);
    return handler(req);
  },
};

export default worker;
