import { Route, Routes } from 'react-router-dom';
import ChatOnlyComponent from './components/ChatBot/ChatOnlyComponent';
import { AuthenticationGuard } from './components/Auth/Auth';
import Home from './Home';
import Dashboard from './components/Dashboard';
import { SKIP_AUTH } from './utils/Constants.ts';
import ThemeWrapper from './context/ThemeWrapper';
import { MessageContextWrapper } from './context/UserMessages';
import { SpotlightProvider } from '@neo4j-ndl/react';
import { FileContextProvider } from './context/UsersFiles';
import UserCredentialsWrapper from './context/UserCredentials';
import AlertContextWrapper from './context/Alert';

const App = () => {
  return (
    <Routes>
      <Route path='/' element={SKIP_AUTH ? <Home /> : <AuthenticationGuard component={Home} />}></Route>
      <Route path='/readonly' element={<Home />}></Route>
      <Route path='/chat-only' element={<ChatOnlyComponent />}></Route>
      <Route
        path='/dashboard'
        element={
          <ThemeWrapper>
            <SpotlightProvider>
              <UserCredentialsWrapper>
                <FileContextProvider>
                  <MessageContextWrapper>
                    <AlertContextWrapper>
                      <Dashboard />
                    </AlertContextWrapper>
                  </MessageContextWrapper>
                </FileContextProvider>
              </UserCredentialsWrapper>
            </SpotlightProvider>
          </ThemeWrapper>
        }
      ></Route>
    </Routes>
  );
};
export default App;
