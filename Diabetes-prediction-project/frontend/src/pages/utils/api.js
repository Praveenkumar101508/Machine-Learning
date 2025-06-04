import axios from 'axios';

const API_URL = 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
});

const handleApiError = (error, customMessage) => {
  console.error(customMessage, error);
  if (error.response) {
    console.error('Response data:', error.response.data);
    console.error('Response status:', error.response.status);
  } else if (error.request) {
    console.error('No response received:', error.request);
  } else {
    console.error('Error setting up request:', error.message);
  }
  throw error;
};

export const uploadBloodSample = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    console.log(`Uploading file: ${file.name}`);
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    console.log('Raw API response:', response);
    console.log('Prediction data:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error uploading blood sample:', error);
    if (error.response) {
      console.error('Error response:', error.response.data);
    }
    throw error;
  }
};

export const fetchDistributionData = async () => {
  try {
    const response = await api.get('/distribution_data');
    return response.data;
  } catch (error) {
    handleApiError(error, 'Error fetching distribution data:');
  }
};

export const fetchFeatureImportance = async () => {
  try {
    const response = await api.get('/feature_importance');
    return response.data;
  } catch (error) {
    console.warn('Error fetching feature importance:', error.message);
    return [
      { name: 'Feature1', importance: 0.2 },
      { name: 'Feature2', importance: 0.2 },
      { name: 'Feature3', importance: 0.2 },
      { name: 'Feature4', importance: 0.2 },
      { name: 'Feature5', importance: 0.2 },
    ];
  }
};

export const fetchCorrelationData = async () => {
  try {
    const response = await api.get('/correlation_data');
    return response.data;
  } catch (error) {
    handleApiError(error, 'Error fetching correlation data:');
    return [];
  }
};

export default api;
