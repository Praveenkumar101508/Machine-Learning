import React, { useEffect, useState } from 'react';
import { Typography, Container, Paper, CircularProgress } from '@mui/material';
import { useLocation } from 'react-router-dom';
import ResultsDisplay from '../components/ResultsDisplay';
import FeatureImportanceChart from '../components/Charts/FeatureImportanceChart';
//import CorrelationHeatmap from '../components/Charts/CorrelationHeatmap';
import { fetchFeatureImportance } from '../utils/api';

const Results = () => {
  const location = useLocation();
  const prediction = location.state?.prediction;
  const [featureImportance, setFeatureImportance] = useState([]);
 // const [correlationData, setCorrelationData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadChartData = async () => {
      try {
        setLoading(true);
        const [importanceData] = await Promise.all([
          fetchFeatureImportance()
          //,fetchCorrelationData()
        ]);
        setFeatureImportance(importanceData);
//        setCorrelationData(corrData);
      } catch (error) {
        console.error('Error loading chart data:', error);
        setError('Failed to load chart data. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    loadChartData();
  }, []);

  useEffect(() => {
    console.log('Prediction data in Results component:', prediction);
  }, [prediction]);

 // const renderCorrelationHeatmap = () => {
 //   if (correlationData && Array.isArray(correlationData) && correlationData.length > 0) {
  //    return <CorrelationHeatmap data={correlationData} />;
   // } else {
   //   return <Typography>No correlation data available.</Typography>;
   // }
 // };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md">
        <Typography color="error" variant="h6">{error}</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md">
      <Typography variant="h2" gutterBottom>
        Diabetes Risk Assessment Results
      </Typography>
     
      {prediction ? (
        <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
          <ResultsDisplay prediction={prediction} />
        </Paper>
      ) : (
        <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" color="error">
            No prediction data available. Please try uploading your blood sample report again.
          </Typography>
        </Paper>
      )}

      <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Feature Importance
        </Typography>
        {featureImportance.length > 0 ? (
          <FeatureImportanceChart data={featureImportance} />
        ) : (
          <Typography>No feature importance data available.</Typography>
        )}
      </Paper>

      
    </Container>
  );
};

export default Results;