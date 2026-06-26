import type { ReactNode } from "react";
import Sidebar from "./Sidebar.tsx";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-y-auto bg-surface-50">
        <div className="flex-1 p-8">{children}</div>
      </main>
    </div>
  );
}
