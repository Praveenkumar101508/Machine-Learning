import React, { useState } from 'react';
import { Typography, Container, Alert } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import UploadForm from '../components/UploadForm';
import { uploadBloodSample } from '../utils/api';

const Upload = () => {
    const navigate = useNavigate();
    const [error, setError] = useState(null);

    const handleUpload = async (file) => {
        try {
            setError(null);
            console.log("Uploading file:", file.name);
            const prediction = await uploadBloodSample(file);
            console.log("Prediction before navigation:", prediction);
            navigate('/results', { state: { prediction } });
        } catch (err) {
            console.error("Error during upload:", err);
            setError("Failed to upload and process the file. Please try again.");
        }
    };

    return (
        <Container maxWidth="md">
            <Typography variant="h2" gutterBottom>
                Upload Blood Sample Report
            </Typography>
            <Typography paragraph>
                Please upload your blood sample report (PDF format) to assess your diabetes risk.
            </Typography>
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}
            <UploadForm onUpload={handleUpload} />
        </Container>
    );
};

export default Upload;