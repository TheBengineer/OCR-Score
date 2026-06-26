import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  FileText,
  Cpu,
  BarChart3,
  type LucideIcon,
} from "lucide-react";

interface NavItem {
  label: string;
  path: string;
  icon: LucideIcon;
}

const navItems: NavItem[] = [
  { label: "Dashboard", path: "/", icon: LayoutDashboard },
  { label: "PDFs", path: "/pdfs", icon: FileText },
  { label: "Engines", path: "/engines", icon: Cpu },
  { label: "Reports", path: "/reports", icon: BarChart3 },
];

export default function Sidebar() {
  return (
    <aside className="flex h-full w-64 flex-col border-r border-surface-200 bg-white">
      {/* Brand */}
      <div className="flex h-16 items-center gap-2 border-b border-surface-200 px-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600 text-sm font-bold text-white">
          O
        </div>
        <span className="text-lg font-semibold text-surface-900">OCRScore</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-primary-50 text-primary-700"
                  : "text-surface-600 hover:bg-surface-100 hover:text-surface-900"
              }`
            }
          >
            <item.icon className="h-5 w-5 shrink-0" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-surface-200 px-6 py-4">
        <p className="text-xs text-surface-400">OCRScore v0.1.0</p>
      </div>
    </aside>
  );
}
