import api from '../API/Index';

export interface AgentMessage {
  type: string;
  content: string;
}

export interface CheckIssuesResponse {
  status: string;
  data: {
    status: string;
    query: string;
    diagnosis?: string;
    display_markdown?: string;
    services?: any[];
    total_services?: number;
    healthy_services?: number;
    unhealthy_services?: number;
    nodedetails?: any;
    sources?: string[];
    entities?: any[];
    model?: string;
    total_tokens?: number;
    response_time?: number;
  };
  message?: string;
}

const checkIssuesAPI = async (): Promise<CheckIssuesResponse | null> => {
  try {
    const response = await api.get<CheckIssuesResponse>('/check-issues');
    console.log('responseeeeeeeeeeeeee=', response);
    return response.data;
  } catch (error) {
    console.error('Error checking issues:', error);
    throw error;
  }
};

export { checkIssuesAPI };
