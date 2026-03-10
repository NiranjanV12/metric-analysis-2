import api from '../API/Index';

export interface ServiceHealth {
  id: string;
  service: string;
  healthUrl: string;
  status: 'Running' | 'Stopped' | 'Unknown';
}

export interface ServiceHealthResponse {
  status: string;
  data: ServiceHealth[];
  message?: string;
}

const getServiceHealthAPI = async (): Promise<ServiceHealth[]> => {
  try {
    const response = await api.get<ServiceHealthResponse>('/get-service-health');
    if (response.data.status === 'Success' && response.data.data) {
      return response.data.data;
    }
    return [];
  } catch (error) {
    console.error('Error fetching service health:', error);
    return [];
  }
};

export { getServiceHealthAPI };
