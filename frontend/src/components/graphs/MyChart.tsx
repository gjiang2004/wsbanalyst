import React from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Label
} from "recharts";
import dayjs from "dayjs";

// Define props type with support for multiple data series
interface MyChartProps {
  chartData: Array<{
    date: string;
    portfolioValue?: number;
    spxValue?: number;
    [key: string]: any;
  }>;
  xKey: string;
  title: string;
  subtitle?: string;
}

const MyChart: React.FC<MyChartProps> = ({ chartData, xKey, title, subtitle }) => {
  // Define colors for the lines
  const portfolioColor = "#4f46e5"; // Indigo
  const spxColor = "#dc2626"; // Red

  // Format value as percentage
  const formatValue = (value: number, isPercentage: boolean) => {
    if (isPercentage) {
      return `${value.toFixed(2)}%`;
    } else {
      return `${(value).toFixed(2)}%`;
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-100">
      {/* Title section with better styling */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-800">{title}</h2>
        {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
        >
          <defs>
            {/* Portfolio gradient */}
            <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={portfolioColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={portfolioColor} stopOpacity={0.3} />
            </linearGradient>

            {/* SPX gradient */}
            <linearGradient id="colorSPX" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={spxColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={spxColor} stopOpacity={0.3} />
            </linearGradient>
          </defs>

          {/* Lighter grid for better readability */}
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />

          {/* Format X-axis with better styling */}
          <XAxis
            dataKey={xKey}
            tickFormatter={(date) => dayjs(date).format("MMM D")}
            angle={-45}
            textAnchor="end"
            interval="preserveStartEnd"
            tick={{ fill: "#666", fontSize: 12 }}
            axisLine={{ stroke: "#e0e0e0" }}
            tickLine={{ stroke: "#e0e0e0" }}
          />

          {/* Format Y-axis with better styling */}
          <YAxis
            tickFormatter={(value) => `${value.toFixed(1)}%`}
            tick={{ fill: "#666", fontSize: 12 }}
            axisLine={{ stroke: "#e0e0e0" }}
            tickLine={{ stroke: "#e0e0e0" }}
          >
            <Label
              value="Percent Change (%)"
              angle={-90}
              position="insideLeft"
              style={{ textAnchor: "middle", fill: "#666", fontSize: 13 }}
              offset={-10}
            />
          </YAxis>

          {/* Enhanced tooltip */}
          <Tooltip
            formatter={(value: number, name: string) => {
              const isPortfolio = name === "WSB";
              return [
                formatValue(value as number, !isPortfolio), 
                name
              ];
            }}
            labelFormatter={(date) => `${dayjs(date).format("MMM D, YYYY")}`}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid #e0e0e0",
              borderRadius: "6px",
              padding: "10px",
              boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
            }}
          />

          {/* Legend with better styling */}
          <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: "12px", color: "#666" }} />

          {/* Portfolio Line */}
          <Line
            type="monotone"
            dataKey="portfolioValue"
            name="WSB"
            stroke={portfolioColor}
            strokeWidth={3}
            dot={{ r: 4, fill: portfolioColor, strokeWidth: 2, stroke: "#fff" }}
            activeDot={{ r: 8, fill: portfolioColor, strokeWidth: 2, stroke: "#fff" }}
            connectNulls={true}
          />

          {/* S&P 500 Line */}
          <Line
            type="monotone"
            dataKey="spxValue"
            name="S&P 500"
            stroke={spxColor}
            strokeWidth={3}
            dot={{ r: 4, fill: spxColor, strokeWidth: 2, stroke: "#fff" }}
            activeDot={{ r: 8, fill: spxColor, strokeWidth: 2, stroke: "#fff" }}
            connectNulls={true}
          />
        </LineChart>
      </ResponsiveContainer>

    </div>
  );
};

export default MyChart;