const { createApp } = Vue

const app = createApp({
    data() {
        return {
            // Dashboard Data
            status: {
                id: 'SUB-001',
                voltage: 0,
                current: 0,
                temperature: 0,
                status: 'Offline',
                anomaly_score: 0,
                prediction: 'Waiting...',
                next_maintenance: 'Loading...'
            },
            monitoring: false,
            timer: null,
            loading: false,
            
            // Chat Data
            chatInput: '',
            chatMessages: [
                { 
                    text: "Hello! I am your Intelligent Maintenance Assistant. I have detailed knowledge about substation equipment maintenance, tools, and safety protocols. How can I help?", 
                    isUser: false, 
                    time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) 
                }
            ],
            isTyping: false,
            
            // Knowledge Base
            showKbModal: false,
            equipmentList: []
        }
    },
    methods: {
        // UI Methods
        async fetchData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                this.status = data;
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        },
        toggleMonitoring() {
            this.monitoring = !this.monitoring;
            if (this.monitoring) {
                this.fetchData();
                this.timer = setInterval(this.fetchData, 2000);
            } else {
                clearInterval(this.timer);
                this.timer = null;
            }
        },
        async resetSystem() {
            this.loading = true;
            setTimeout(() => {
                this.status.voltage = 0;
                this.status.current = 0;
                this.status.temperature = 0;
                this.status.prediction = 'System Reset';
                this.loading = false;
                this.addSystemMessage("System reset sequence completed.");
            }, 1500);
        },
        getPredictionColor(prediction) {
            if (prediction === 'Normal') return 'text-green-400';
            if (prediction === 'Warning') return 'text-yellow-400';
            return 'text-red-500';
        },

        // Chat Methods
        async sendMessage() {
            if (!this.chatInput.trim()) return;
            
            const userMsg = this.chatInput;
            this.addMessage(userMsg, true);
            this.chatInput = '';
            this.isTyping = true;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: userMsg })
                });
                const data = await response.json();
                
                setTimeout(() => {
                    this.addMessage(data.response, false, data.accuracy);
                    this.isTyping = false;
                }, 500);
                
            } catch (error) {
                console.error('Chat error:', error);
                this.addMessage("Sorry, I encountered an error connecting to the server.", false, 0);
                this.isTyping = false;
            }
        },
        addMessage(text, isUser, accuracy = null) {
            this.chatMessages.push({
                text: text,
                isUser: isUser,
                accuracy: accuracy,
                time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            });
            this.$nextTick(() => {
                this.scrollToBottom();
            });
        },
        addSystemMessage(text) {
            this.chatMessages.push({
                text: `[SYSTEM] ${text}`,
                isUser: false,
                time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            });
            this.$nextTick(() => {
                this.scrollToBottom();
            });
        },
        quickQuery(text) {
            this.chatInput = text;
            this.sendMessage();
        },
        scrollToBottom() {
            const container = this.$refs.chatContainer;
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        },
        formatMessage(text) {
            // Simple formatter to handle newlines and bold text from bot
            let formatted = text.replace(/\n/g, '<br>');
            formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            return formatted;
        },

        // Knowledge Base Methods
        async openKbModal() {
            this.showKbModal = true;
            if (this.equipmentList.length === 0) {
                try {
                    const response = await fetch('/api/equipment');
                    this.equipmentList = await response.json();
                } catch (error) {
                    console.error('Error fetching equipment list:', error);
                }
            }
        },
        selectEquipment(name) {
            this.showKbModal = false;
            this.chatInput = `Tell me about ${name}`;
            this.sendMessage();
        },

        // Chat History Methods
        clearChat() {
            this.chatMessages = [{
                text: "Chat cleared. Ask me a new question about substation equipment.",
                isUser: false,
                time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            }];
        },
        exportChat() {
            let chatHistory = "Substation Bot Chat History\n";
            chatHistory += "============================\n\n";
            this.chatMessages.forEach(msg => {
                const prefix = msg.isUser ? "User" : "Bot";
                const accuracy = msg.accuracy ? ` (Accuracy: ${msg.accuracy}%)` : "";
                chatHistory += `[${msg.time}] ${prefix}:${accuracy}\n${msg.text}\n\n`;
            });

            const blob = new Blob([chatHistory], { type: 'text/plain' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `substation-chat-history-${new Date().toISOString().split('T')[0]}.txt`;
            link.click();
            URL.revokeObjectURL(link.href);
        }
    },
    mounted() {
        this.fetchData();
    }
});

app.mount('#app');
