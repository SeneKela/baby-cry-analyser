import { motion } from "motion/react";
import { X } from "lucide-react";

interface ResultCardProps {
    result: {
        type: string;
        confidence: number;
        description: string;
        suggestions: string[];
        color: string;
        icon: string;
    };
    onClose: () => void;
}

export function ResultCard({ result, onClose }: ResultCardProps) {
    return (
        <>
            {/* Backdrop */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
                onClick={onClose}
            />

            {/* Card */}
            <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{
                    opacity: 1,
                    scale: 1,
                    y: 0,
                    transition: {
                        type: "spring",
                        damping: 20,
                        stiffness: 300
                    }
                }}
                exit={{
                    opacity: 0,
                    scale: 0.95,
                    y: 10,
                    transition: {
                        duration: 0.15
                    }
                }}
                className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-md z-50"
                style={{ maxHeight: '85vh' }}
            >
                <div className="bg-white rounded-3xl shadow-2xl overflow-hidden">
                    {/* Header */}
                    <div
                        className="relative px-6 py-6"
                        style={{
                            background: `linear-gradient(135deg, ${result.color}20 0%, ${result.color}05 100%)`
                        }}
                    >
                        <button
                            onClick={onClose}
                            className="absolute right-4 top-4 p-2 rounded-full bg-white/80 hover:bg-white transition-colors"
                        >
                            <X className="size-4 text-gray-600" />
                        </button>

                        <div className="flex items-center gap-4 mb-4">
                            <div
                                className="size-14 rounded-2xl flex items-center justify-center text-3xl shadow-md"
                                style={{
                                    backgroundColor: `${result.color}30`,
                                }}
                            >
                                {result.icon}
                            </div>
                            <div className="flex-1">
                                <h2 className="text-2xl font-bold text-gray-900">
                                    {result.type}
                                </h2>
                                <p className="text-sm text-gray-600 mt-0.5">Analysis Complete</p>
                            </div>
                        </div>

                        {/* Confidence */}
                        <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-600">Confidence</span>
                                <span
                                    className="font-semibold px-2.5 py-0.5 rounded-full text-sm"
                                    style={{
                                        backgroundColor: `${result.color}20`,
                                        color: result.color
                                    }}
                                >
                                    {result.confidence}%
                                </span>
                            </div>
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${result.confidence}%` }}
                                    transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                                    className="h-full rounded-full"
                                    style={{
                                        background: `linear-gradient(90deg, ${result.color} 0%, ${result.color}dd 100%)`
                                    }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Content */}
                    <div className="px-6 py-5 max-h-[50vh] overflow-y-auto">
                        {/* Description */}
                        <div className="mb-5">
                            <p className="text-gray-700 leading-relaxed">
                                {result.description}
                            </p>
                        </div>

                        {/* Suggestions */}
                        {result.suggestions.length > 0 && (
                            <div>
                                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                    What to do
                                    <span
                                        className="size-1.5 rounded-full"
                                        style={{ backgroundColor: result.color }}
                                    />
                                </h4>
                                <ul className="space-y-2.5">
                                    {result.suggestions.map((suggestion, index) => (
                                        <motion.li
                                            key={index}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: 0.3 + (index * 0.1) }}
                                            className="flex items-start gap-2.5"
                                        >
                                            <span
                                                className="size-1.5 rounded-full mt-2 flex-shrink-0"
                                                style={{ backgroundColor: result.color }}
                                            />
                                            <span className="text-sm text-gray-600 leading-relaxed">
                                                {suggestion}
                                            </span>
                                        </motion.li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-6 pb-6">
                        <button
                            onClick={onClose}
                            className="w-full py-3.5 rounded-2xl text-white font-medium transition-all"
                            style={{
                                backgroundColor: result.color,
                                boxShadow: `0 4px 12px ${result.color}40`
                            }}
                        >
                            Got it
                        </button>
                    </div>
                </div>
            </motion.div>
        </>
    );
}
