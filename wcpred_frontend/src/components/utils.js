function getBaseURL() {
    const envURL = process.env.WCPRED_BASE_URL;
    if (envURL !== undefined) return envURL;
    return "http://localhost:5000";
  }

  export default getBaseURL;