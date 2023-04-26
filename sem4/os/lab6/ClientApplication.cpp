#include "Client.hpp"
#include <TGUI/TGUI.hpp>
#include <SFML/System.hpp>
#include <thread>
#include <functional>
#include <future>

class ClientApplication
{
	Client m_client;

	sf::RenderWindow m_window;

	tgui::GuiSFML m_gui;

public:

	ClientApplication(unsigned int width, unsigned int height) :
		m_window({ width, height }, "Chat"), m_gui(m_window)
	{
		m_window.setFramerateLimit(60);
		tgui::Theme::setDefault("themes\\Black.txt");
		turnToRegistration();
	}

	void run()
	{
		registry();
		turnToChat();
		waitConnection();
		if (m_client.isInChat())
		{
			startChating();
		}
	}

private:

	void startChating()
	{
		auto updateChatBox = [this](const Message& message) {
			m_gui.get<tgui::ChatBox>("Chat")->addLine(message.toString());
		};
		while (m_window.isOpen() && m_client.isInChat())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				switch (event.type)
				{
				case sf::Event::Closed:
					m_window.close();
					break;
				default:
					break;
				}
				m_gui.handleEvent(event);
			}
			if (m_client.isDirty())
			{
				m_gui.get<tgui::ChatBox>("Chat")->removeAllLines();
				m_client.processChat(updateChatBox);
			}
			m_window.clear();
			m_gui.draw();
			m_window.display();
		}
		if (!m_client.isInChat())
		{
			tgui::MessageBox::Ptr waitingBox = tgui::MessageBox::create("Info", "The connection to the server was lost");
			waitingBox->setOrigin(0.5f, 0.5f);
			waitingBox->setPosition("50%", "50%");
			m_gui.add(waitingBox);
			m_gui.mainLoop();
		}
	}

	void registry()
	{
		while (!m_client.isAvailableName() && m_window.isOpen())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				switch (event.type)
				{
				case sf::Event::Closed:
					m_window.close();
					break;
				default:
					break;
				}
				m_gui.handleEvent(event);
			}
			m_window.clear();
			m_gui.draw();
			m_window.display();
		}
	}

	void waitConnection()
	{
		std::atomic_bool success = true;
		std::thread{[&success, this]() {
			success = m_client.join();
		}}.detach();
		tgui::MessageBox::Ptr waitingBox = tgui::MessageBox::create("Connect", "Waiting for a chat connection...");
		waitingBox->setOrigin(0.5f, 0.5f);
		waitingBox->setPosition("50%", "50%");
		m_gui.add(waitingBox);
		while (m_window.isOpen() && success && !m_client.isInChat())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				switch (event.type)
				{
				case sf::Event::Closed:
					m_window.close();
					break;
				default:
					break;
				}
				m_gui.handleEvent(event);
			}
			m_window.clear();
			m_gui.draw();
			m_window.display();
		}
		if (!m_window.isOpen())
		{
			return;
		}
		if (success)
		{
			turnToChat();
		}
		else
		{
			waitingBox->setText("Failed to connect to chat");
			m_gui.mainLoop();
		}
	}

	void turnToChat()
	{
		m_gui.removeAllWidgets();

		tgui::ChatBox::Ptr chatBox = tgui::ChatBox::create();
		chatBox->setPosition("5%", "5%");
		chatBox->setSize("90%", "70%");
		m_gui.add(chatBox, "Chat");

		tgui::EditBox::Ptr inputField = tgui::EditBox::create();
		inputField->setPosition("5%", "80%");
		inputField->setSize("60%", "15%");
		m_gui.add(inputField, "Input");

		tgui::Button::Ptr buttonSend = tgui::Button::create("Send");
		buttonSend->setPosition("70%", "80%");
		// buttonSend->setSize(200, 80);
		buttonSend->onClick([this, inputField]() mutable {
			std::string content = inputField->getText().toStdString();
			if (!content.empty())
			{
				m_client.sendMessage(content);
				inputField->setText("");
			}
			});
		m_gui.add(buttonSend, "Send");

		tgui::Button::Ptr buttonLeave = tgui::Button::create("Leave");
		buttonLeave->setPosition("70%", "90%");
		// buttonSend->setSize(200, 80);
		buttonLeave->onClick([this]() mutable {
			m_client.detach();
			});
		m_gui.add(buttonLeave, "Leave");
	}

	void turnToRegistration()
	{
		m_gui.removeAllWidgets();

		tgui::Label::Ptr inviteLabel = tgui::Label::create("Enter your username:");
		inviteLabel->setOrigin(0.5f, 0.5f);
		inviteLabel->setPosition("50%", "30%");
		m_gui.add(inviteLabel);

		tgui::EditBox::Ptr inputField = tgui::EditBox::create();
		inputField->setOrigin(0.5f, 0.5f);
		inputField->setPosition("50%", "50%");
		inputField->setInputValidator(R"(\b[a-zA-Z0-9]{0,8}\b)");
		m_gui.add(inputField);

		tgui::Button::Ptr buttonOk = tgui::Button::create("Ok");
		buttonOk->setOrigin(0.5f, 0.f);
		buttonOk->setPosition("50%", "80%");
		buttonOk->onClick([this, inputField]() mutable {
			m_client.setUsername(inputField->getText().toStdString());
			});
		m_gui.add(buttonOk);
	}
};



int main(int argc, char const* argv[])
{
	ClientApplication app(800, 600);
	app.run();
}
