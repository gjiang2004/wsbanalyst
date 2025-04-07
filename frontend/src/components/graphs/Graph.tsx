import React, { useState, useEffect, useRef } from "react";
import MyChart from "./MyChart";
import sampleData from "../../portfolio_data/portfolio_total_investment.json";
import spxData from "../../portfolio_data/spx_data.json";
import './Graph.css';

// Define the expected structure of the portfolio data
interface PortfolioEntry {
    date: string;
    investment: number;
    today_profit: number;
    total_profit: number;
    percentchange?: number; // This will be added in useEffect
}

interface SPXEntry {
    Date: string;
    "Percent Change": number;
}

interface CombinedDataPoint {
    date: string;
    portfolioValue?: number;
    spxValue?: number;
}

const Graph: React.FC = () => {
    const [portfolioData, setPortfolioData] = useState<PortfolioEntry[]>([]);
    const [comparisonData, setComparisonData] = useState<CombinedDataPoint[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);

    useEffect(() => {
        try {
            // Process portfolio data
            const processedData: PortfolioEntry[] = sampleData.portfolio_statistics.map((entry: PortfolioEntry) => ({
                ...entry,
                percentchange: entry.investment ? entry.total_profit / entry.investment : 0, // Avoid division by zero
            }));
            setPortfolioData(processedData);

            // Map of dates to combined data points
            const combinedDataMap = new Map<string, CombinedDataPoint>();

            // Process portfolio data
            processedData.forEach(entry => {
                combinedDataMap.set(entry.date, {
                    date: entry.date,
                    portfolioValue: entry.percentchange ? entry.percentchange * 100 : 0, // Convert to percentage
                });
            });

            // Process S&P 500 data and merge with portfolio data
            spxData.forEach((entry: SPXEntry) => {
                const date = entry.Date;
                const existing = combinedDataMap.get(date) || { date };
                
                combinedDataMap.set(date, {
                    ...existing,
                    spxValue: entry["Percent Change"]
                });
            });

            // Convert map to array and sort by date
            const combinedArray = Array.from(combinedDataMap.values())
                .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

            console.log("Combined data points:", combinedArray.length);
            console.log("Sample data point:", combinedArray[0]);
            
            setComparisonData(combinedArray);
        } catch (error) {
            console.error("Error processing data:", error);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Calculate summary statistics
    const getStats = () => {
        if (!portfolioData.length) return null;

        const latestEntry = portfolioData[portfolioData.length - 1];
        
        // Calculate overall growth
        const overallChange = latestEntry.percentchange || 0;
        const overallChangePercent = (overallChange * 100).toFixed(2);
        
        // Format the investment amount with commas and decimal places
        const currentInvestment = latestEntry.investment.toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });

        // Format total profit
        const totalProfit = latestEntry.total_profit.toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });

        // Calculate S&P 500 performance if data is available
        let spxPerformance = "N/A";
        if (spxData && spxData.length > 0) {
            const latestSPX = spxData[spxData.length - 1];
            spxPerformance = `${(latestSPX["Percent Change"]).toFixed(2)}%`;
        }

        return { overallChangePercent, currentInvestment, totalProfit, spxPerformance };
    };

    const stats = getStats();

    return (
        <div className="bg-gray-50 p-6 rounded-lg">
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-800 mb-2" style={{paddingBottom: "4vh"}}>Portfolio Performance Dashboard</h1>
            </div>

        <div className="grid-container">
        {isLoading ? (
            <div className="flex justify-center items-center h-64 bg-white rounded-lg border border-gray-200">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="h-4 w-24 bg-gray-200 rounded mb-3"></div>
                    <div className="h-32 w-full max-w-md bg-gray-200 rounded"></div>
                </div>
            </div>
        ) : portfolioData.length ? (
            <>
                {/* Stats Summary Cards */}
                {stats && (
                    <div className="stats-container">
                        <div className="stat-card">
                        <p>Total Investment</p>
                        <p>{stats.currentInvestment}</p>
                        </div>
                        <div className="stat-card">
                        <p>Total Profit</p>
                        <p>{stats.totalProfit}</p>
                        </div>
                        <div className="stat-card">
                        <p>Your ROI</p>
                        <p className={parseFloat(stats.overallChangePercent) >= 0 ? 'positive' : 'negative'}>
                            {stats.overallChangePercent}%
                        </p>
                        </div>
                        <div className="stat-card">
                        <p>S&P 500 Return</p>
                        <p className={stats.spxPerformance !== "N/A" && parseFloat(stats.spxPerformance) >= 0 ? 'positive' : 'negative'}>
                            {stats.spxPerformance}
                        </p>
                        </div>
                    </div>
                    )}

                    <div className="explanation">
                        <p>The total investment for our portfolio is calculated by adding the amount from buying and shorting stocks. Our percentage gain is calculated if we sold and bought back all stocks traded in the last 30 days, starting from the initial price. Stocks are traded based on post sentiment and upvotes.</p>
                    </div>
                    {/* Main Chart */}
                    {comparisonData.length > 0 && (
                    <div className="chart-section">
                        <MyChart chartData={comparisonData} xKey="date" title="Performance Comparison" subtitle="WSB vs. S&P 500" />
                    </div>
                    )}
                    
            </>
        ) : (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 text-center">
                <p className="text-gray-600">No portfolio data available</p>
            </div>
        )}
        </div>
        </div>
    );
};

export default Graph;