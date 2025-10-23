import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import type { InferenceResult } from '../types';

export const useResults = () => {
  const [results, setResults] = useState<InferenceResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getResults();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  return { results, loading, error, refetch: fetchResults };
};

export const useResult = (resultId: string | null) => {
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId) {
      setResult(null);
      return;
    }

    const fetchResult = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await apiService.getResult(resultId);
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch result');
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [resultId]);

  return { result, loading, error };
};

