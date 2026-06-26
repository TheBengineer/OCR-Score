import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.tsx";
import Dashboard from "./pages/Dashboard.tsx";
import PdfList from "./pages/PdfList.tsx";
import PdfViewer from "./pages/PdfViewer.tsx";
import Evaluation from "./pages/Evaluation.tsx";
import EngineList from "./pages/EngineList.tsx";
import Reports from "./pages/Reports.tsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/pdfs" element={<PdfList />} />
        <Route path="/pdfs/:id" element={<PdfViewer />} />
        <Route path="/pdfs/:id/evaluation" element={<Evaluation />} />
        <Route path="/engines" element={<EngineList />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="*" element={<Dashboard />} />
      </Routes>
    </Layout>
  );
}
