// frontend/src/pages/Home.js

import React, { useEffect, useState } from 'react';
import { Typography, Container, Button, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import Introduction from '../components/Introduction';
import DistributionChart from '../components/Charts/DistributionChart';
import { fetchDistributionData } from '../utils/api';

const Home = () => {
  const [distributionData, setDistributionData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDistributionData = async () => {
      try {
        setLoading(true);
        const data = await fetchDistributionData();
        setDistributionData(data);
      } catch (error) {
        console.error('Failed to load distribution data:', error);
        setError('Failed to load distribution data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    loadDistributionData();
  }, []);

  return (
    <Container maxWidth="md">
      <Typography variant="h2" gutterBottom>
        Welcome to Diabetic Detection
      </Typography>
      <Introduction />
      <Typography variant="h4" gutterBottom>
        Diabetes Distribution in Our Dataset
      </Typography>
      {loading ? (
        <CircularProgress />
      ) : error ? (
        <Typography color="error">{error}</Typography>
      ) : (
        <DistributionChart data={distributionData} />
      )}
      <Button
        variant="contained"
        color="primary"
        component={Link}
        to="/upload"
        size="large"
        sx={{ marginTop: 2 }}
      >
        Get Started
      </Button>
    </Container>
  );
};

export default Home;